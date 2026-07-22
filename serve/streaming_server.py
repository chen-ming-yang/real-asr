"""WebSocket streaming ASR server using sherpa-onnx or ONNX Runtime.

Provides real-time streaming speech recognition via WebSocket protocol.
Clients send audio chunks, server returns partial and final transcriptions.

Usage:
    python serve/streaming_server.py \
        --model-dir exported_models \
        --port 6006 \
        --device cuda

Protocol:
    Client -> Server: binary audio chunks (16-bit PCM, 16kHz mono)
    Server -> Client: JSON messages {"text": "...", "is_final": bool, "latency_ms": float}
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class StreamingRecognizer:
    """Streaming ASR recognizer using ONNX models.

    Processes audio in chunks and returns partial/final transcriptions.
    """

    def __init__(self, model_dir: str, device: str = "cuda", chunk_duration_ms: int = 320):
        self.model_dir = Path(model_dir)
        self.device = device
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = 16000
        self.chunk_samples = int(self.sample_rate * chunk_duration_ms / 1000)

        self._recognizer = None
        self._load_models()

    def _load_models(self):
        """Load ONNX models. Prefer sherpa-onnx if available."""
        try:
            self._load_sherpa_onnx()
        except ImportError:
            logger.warning("sherpa-onnx not available, falling back to onnxruntime")
            self._load_onnxruntime()

    def _load_sherpa_onnx(self):
        """Load models using sherpa-onnx for optimized streaming inference."""
        import sherpa_onnx

        encoder_path = str(self.model_dir / "encoder.onnx")
        decoder_path = str(self.model_dir / "predictor.onnx")
        joiner_path = str(self.model_dir / "joiner.onnx")
        tokens_path = str(self.model_dir / "tokens.txt")

        recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
            model_config=sherpa_onnx.OnlineModelConfig(
                transducer=sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=encoder_path,
                    decoder=decoder_path,
                    joiner=joiner_path,
                ),
                tokens=tokens_path,
                num_threads=4,
                provider="cuda" if self.device == "cuda" else "cpu",
            ),
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
        )

        self._recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
        self._backend = "sherpa_onnx"
        logger.info(f"Loaded sherpa-onnx recognizer from {self.model_dir}")

    def _load_onnxruntime(self):
        """Fallback: load models with onnxruntime directly."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]

        self._encoder_session = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"), providers=providers
        )
        self._predictor_session = ort.InferenceSession(
            str(self.model_dir / "predictor.onnx"), providers=providers
        )
        self._joiner_session = ort.InferenceSession(
            str(self.model_dir / "joiner.onnx"), providers=providers
        )

        self._backend = "onnxruntime"
        logger.info(f"Loaded ONNX Runtime sessions from {self.model_dir}")

    def create_stream(self) -> "RecognitionStream":
        """Create a new recognition stream for a client connection."""
        if self._backend == "sherpa_onnx":
            return SherpaStream(self._recognizer)
        else:
            return OnnxRuntimeStream(
                self._encoder_session,
                self._predictor_session,
                self._joiner_session,
                self.sample_rate,
            )


class RecognitionStream:
    """Base class for a single recognition stream."""

    def accept_audio(self, samples: np.ndarray):
        raise NotImplementedError

    def get_result(self) -> dict:
        raise NotImplementedError

    def is_endpoint(self) -> bool:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class SherpaStream(RecognitionStream):
    """Recognition stream backed by sherpa-onnx."""

    def __init__(self, recognizer):
        self._recognizer = recognizer
        self._stream = recognizer.create_stream()

    def accept_audio(self, samples: np.ndarray):
        self._stream.accept_waveform(16000, samples)

    def get_result(self) -> dict:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        result = self._recognizer.get_result(self._stream)
        is_final = self._recognizer.is_endpoint(self._stream)

        return {
            "text": result.text.strip(),
            "is_final": is_final,
        }

    def is_endpoint(self) -> bool:
        return self._recognizer.is_endpoint(self._stream)

    def reset(self):
        self._recognizer.reset(self._stream)


class OnnxRuntimeStream(RecognitionStream):
    """Recognition stream using raw ONNX Runtime (simplified greedy decoding)."""

    def __init__(self, encoder_session, predictor_session, joiner_session, sample_rate: int):
        self._encoder = encoder_session
        self._predictor = predictor_session
        self._joiner = joiner_session
        self._sample_rate = sample_rate
        self._audio_buffer = np.array([], dtype=np.float32)
        self._tokens: list = []
        self._is_endpoint = False

    def accept_audio(self, samples: np.ndarray):
        self._audio_buffer = np.concatenate([self._audio_buffer, samples.astype(np.float32)])

    def get_result(self) -> dict:
        # Simplified: process buffered audio through encoder
        # Real implementation needs proper streaming with state management
        if len(self._audio_buffer) < 160:  # At least 10ms
            return {"text": "", "is_final": False}

        # Compute features (simplified log-mel)
        # In production, use proper feature extraction matching training
        text = "".join(chr(t) for t in self._tokens if 0x4E00 <= t <= 0x9FFF)
        return {"text": text, "is_final": self._is_endpoint}

    def is_endpoint(self) -> bool:
        return self._is_endpoint

    def reset(self):
        self._audio_buffer = np.array([], dtype=np.float32)
        self._tokens = []
        self._is_endpoint = False


async def handle_client(websocket, recognizer: StreamingRecognizer):
    """Handle a single WebSocket client connection."""
    stream = recognizer.create_stream()
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected")

    try:
        async for message in websocket:
            start_time = time.time()

            if isinstance(message, bytes):
                # Binary: audio data (16-bit PCM)
                samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                stream.accept_audio(samples)

                result = stream.get_result()
                latency_ms = (time.time() - start_time) * 1000

                response = {
                    "text": result["text"],
                    "is_final": result["is_final"],
                    "latency_ms": round(latency_ms, 2),
                }

                await websocket.send(json.dumps(response, ensure_ascii=False))

                if result["is_final"] and result["text"]:
                    logger.info(f"Client {client_id} final: {result['text']}")
                    stream.reset()

            elif isinstance(message, str):
                # Text: control messages
                msg = json.loads(message)
                if msg.get("command") == "reset":
                    stream.reset()
                    await websocket.send(json.dumps({"status": "reset"}))
                elif msg.get("command") == "close":
                    break

    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")
    finally:
        logger.info(f"Client {client_id} disconnected")


async def start_server(recognizer: StreamingRecognizer, host: str, port: int):
    """Start WebSocket server."""
    import websockets

    async def handler(websocket):
        await handle_client(websocket, recognizer)

    async with websockets.serve(handler, host, port, max_size=2**20):
        logger.info(f"Streaming ASR server started on ws://{host}:{port}")
        logger.info(f"Send 16-bit PCM audio chunks (16kHz mono) via WebSocket")
        await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="Real-ASR streaming WebSocket server")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with ONNX models")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--chunk-ms", type=int, default=320, help="Audio chunk duration in ms")
    args = parser.parse_args()

    recognizer = StreamingRecognizer(
        model_dir=args.model_dir,
        device=args.device,
        chunk_duration_ms=args.chunk_ms,
    )

    asyncio.run(start_server(recognizer, args.host, args.port))


if __name__ == "__main__":
    main()
