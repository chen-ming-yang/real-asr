"""SentencePiece tokenizer training for Real-ASR.

Trains a unigram SentencePiece model on transcription text data.
Vocabulary is designed for Chinese dysarthric ASR (character-dominant with some BPE).

Usage:
    python data/train_tokenizer.py \
        --text-file data/all_transcripts.txt \
        --vocab-size 6000 \
        --output-dir tokenizer
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_texts_from_manifests(manifest_dirs: list, output_path: Path):
    """Collect all transcription texts from Lhotse manifests."""
    texts = []
    for manifest_dir in manifest_dirs:
        sup_path = Path(manifest_dir) / "supervisions.jsonl"
        if not sup_path.exists():
            continue
        with open(sup_path, "r", encoding="utf-8") as f:
            for line in f:
                sup = json.loads(line)
                text = sup.get("text", "").strip()
                if text:
                    texts.append(text)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    logger.info(f"Extracted {len(texts)} texts to {output_path}")
    return output_path


def train_sentencepiece(
    text_file: str,
    output_dir: str,
    vocab_size: int = 6000,
    model_type: str = "unigram",
    character_coverage: float = 0.9999,
):
    """Train SentencePiece tokenizer."""
    import sentencepiece as spm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = str(output_dir / "tokenizer")

    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        input_sentence_size=5000000,
        shuffle_input_sentence=True,
        # Reserve special tokens
        pad_id=0,  # blank for CTC/RNN-T
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Chinese-specific settings
        byte_fallback=True,
        split_by_unicode_script=True,
        split_digits=True,
        treat_whitespace_as_suffix=True,
    )

    logger.info(f"Trained tokenizer: {model_prefix}.model (vocab={vocab_size})")

    # Generate tokens.txt for sherpa-onnx compatibility
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    tokens_path = output_dir / "tokens.txt"
    with open(tokens_path, "w", encoding="utf-8") as f:
        for i in range(sp.GetPieceSize()):
            piece = sp.IdToPiece(i)
            f.write(f"{piece} {i}\n")

    logger.info(f"Saved tokens.txt to {tokens_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--text-file", type=str, help="Path to text file (one sentence per line)")
    parser.add_argument("--manifest-dirs", nargs="+", default=[], help="Lhotse manifest directories")
    parser.add_argument("--output-dir", type=str, default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe", "char"])
    args = parser.parse_args()

    text_file = args.text_file
    if not text_file and args.manifest_dirs:
        text_file = str(Path(args.output_dir) / "all_texts.txt")
        extract_texts_from_manifests(args.manifest_dirs, Path(text_file))

    if not text_file:
        parser.error("Must provide either --text-file or --manifest-dirs")

    train_sentencepiece(
        text_file=text_file,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
