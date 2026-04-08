from __future__ import annotations

import argparse
from pathlib import Path

from ddsp_piano_pytorch.data_pipeline import preprocess_data_into_tfrecord


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO dataset into PyTorch manifest format.")
    parser.add_argument("-sr", "--sample_rate", type=int, default=24000)
    parser.add_argument("-fr", "--frame_rate", type=int, default=250)
    parser.add_argument("-p", "--polyphony", type=int, default=16)
    parser.add_argument("maestro_dir", type=str)
    parser.add_argument("out_dir", type=str)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess_data_into_tfrecord(
        out_dir / "maestro_validation.tfrecord",
        dataset_dir=args.maestro_dir,
        split="validation",
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        max_polyphony=args.polyphony,
    )
    preprocess_data_into_tfrecord(
        out_dir / "maestro_train.tfrecord",
        dataset_dir=args.maestro_dir,
        split="train",
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        max_polyphony=args.polyphony,
    )


if __name__ == "__main__":
    main(process_args())
