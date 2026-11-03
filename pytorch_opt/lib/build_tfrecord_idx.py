#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
import struct


def iter_record_offsets(path: Path):
    offset = 0
    with path.open('rb') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) == 0:
                break
            if len(hdr) != 8:
                raise RuntimeError(f"Corrupted TFRecord (short header) at offset {offset} in {path}")
            length = struct.unpack('<Q', hdr)[0]
            yield offset
            f.seek(4 + length + 4, os.SEEK_CUR)
            offset = f.tell()


def build_index_for(path: Path, overwrite: bool = False) -> Path | None:
    if path.suffix.lower() != '.tfrec':
        return None
    idx_path = Path(str(path) + '.idx')
    if idx_path.exists() and not overwrite:
        return None
    idx_path_tmp = idx_path.with_suffix(idx_path.suffix + '.tmp')
    idx_path_tmp.parent.mkdir(parents=True, exist_ok=True)
    with idx_path_tmp.open('w', encoding='utf-8') as w:
        for off in iter_record_offsets(path):
            w.write(f"{off}\n")
    idx_path_tmp.replace(idx_path)
    return idx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Directory with *.tfrec files')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing .idx files')
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"[build-idx] Directory not found: {root}")

    patterns = [
        'train*.tfrec', 'test*.tfrec', 'val*.tfrec', 'valid*.tfrec'
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))

    if not files:
        raise SystemExit(f"[build-idx] No *.tfrec files found under {root}")

    created = 0
    skipped = 0
    for f in files:
        out = build_index_for(f, overwrite=args.overwrite)
        if out is None:
            skipped += 1
        else:
            created += 1
            print(f"[build-idx] wrote {out}")
    print(f"[build-idx] done. created={created} skipped={skipped}")


if __name__ == '__main__':
    main()
