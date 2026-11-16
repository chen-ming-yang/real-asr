"""Extract text from lhotse supervision manifests for LM training.

Reads supervisions.jsonl, extracts text, deduplicates, tokenizes with
SentencePiece, and writes token ID sequences for LM training.

Usage:
    python scripts/extract_lm_text.py \
        --manifest data/manifests/wenetspeech/train_ms/supervisions.jsonl \
        --tokenizer data/tokenizer/sp_6000.model \
        --output data/lm_train_tokens.txt \
        --output-text data/lm_train_text.txt
"""

import argparse
import json
import logging
from pathlib import Path

import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract LM training text from manifests")
    parser.add_argument("--manifest", type=str, nargs="+", required=True,
                        help="Path(s) to supervisions.jsonl files")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="SentencePiece model path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output token IDs file (one sentence per line, space-separated IDs)")
    parser.add_argument("--output-text", type=str, default=None,
                        help="Output raw text file (for inspection)")
    parser.add_argument("--min-length", type=int, default=2,
                        help="Minimum text length in characters")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Maximum text length in characters")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    logger.info(f"Loaded tokenizer: vocab_size={sp.GetPieceSize()}")

    texts = set()
    total_read = 0

    for manifest_path in args.manifest:
        logger.info(f"Reading {manifest_path}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                total_read += 1
                item = json.loads(line)
                text = item.get("text", "").strip()
                if len(text) < args.min_length or len(text) > args.max_length:
                    continue
                texts.add(text)

                if total_read % 500000 == 0:
                    logger.info(f"  Read {total_read} lines, unique texts: {len(texts)}")

    logger.info(f"Total read: {total_read}, unique texts: {len(texts)}")

    texts_list = sorted(texts)

    if args.output_text:
        Path(args.output_text).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_text, "w", encoding="utf-8") as f:
            for t in texts_list:
                f.write(t + "\n")
        logger.info(f"Wrote raw text to {args.output_text}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in texts_list:
            ids = sp.Encode(t, out_type=int)
            f.write(" ".join(map(str, ids)) + "\n")

    logger.info(f"Wrote {len(texts_list)} tokenized sentences to {args.output}")
    logger.info(f"Done.")


if __name__ == "__main__":
    main()
