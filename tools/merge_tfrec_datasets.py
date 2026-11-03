#!/usr/bin/env python3
import argparse
import collections
import glob
import os
import random
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_val_split import parse_example
from make_tfrec_idx import record_offsets

SPLITS = ("train", "val", "test")


def index_source(tag, d, split):
    out = []
    for path in sorted(glob.glob(os.path.join(d, f"{split}*.tfrec"))):
        with open(path, "rb") as fh:
            raw = fh.read()
        for off in record_offsets(path):
            (ln,) = struct.unpack("<Q", raw[off:off + 8])
            ex = parse_example(raw[off + 12:off + 12 + ln])
            out.append((tag, path, off, ln + 16,
                        ex["image_name"][1].decode(), ex["retinopatia"][1]))
    return out


def _write_shard(task):
    out_path, entries = task
    by_file = collections.defaultdict(list)
    for i, (_, path, off, size, _, _) in enumerate(entries):
        by_file[path].append((i, off, size))

    blocks = [None] * len(entries)
    for path, items in by_file.items():
        with open(path, "rb") as fh:
            for i, off, size in items:
                fh.seek(off)
                blocks[i] = fh.read(size)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as w:
        for b in blocks:
            w.write(b)
    os.replace(tmp, out_path)

    offsets = record_offsets(out_path)
    if len(offsets) != len(entries):
        raise RuntimeError(f"{out_path}: {len(entries)} registros gravados, "
                           f"indice achou {len(offsets)}")
    itmp = out_path + ".idx.tmp"
    with open(itmp, "w") as w:
        w.write("".join(f"{o}\n" for o in offsets))
    os.replace(itmp, out_path + ".idx")
    return len(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True,
                    metavar="TAG=DIR", help="pode repetir; ex: --src hcpa=data/all-tfrec-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    sources = []
    for s in args.src:
        if "=" not in s:
            sys.exit(f"--src precisa ser TAG=DIR (recebi {s})")
        tag, d = s.split("=", 1)
        if not os.path.isdir(d):
            sys.exit(f"nao encontrei {d}")
        sources.append((tag, d))

    os.makedirs(args.out, exist_ok=True)
    manifest = open(os.path.join(args.out, "provenance.csv"), "w")
    manifest.write("split,shard,imagem,retinopatia,dataset\n")

    seen = {}
    grand = 0
    print(f"{'split':6s} " + " ".join(f"{t:>9s}" for t, _ in sources) +
          f" {'total':>9s} {'prev':>6s} {'shards':>7s}")
    for split in SPLITS:
        entries = []
        per_src = {}
        for tag, d in sources:
            got = index_source(tag, d, split)
            per_src[tag] = len(got)
            entries.extend(got)
        if not entries:
            continue

        for e in entries:
            if e[4] in seen:
                sys.exit(f"COLISAO de nome: {e[4]} em {e[0]} e em {seen[e[4]]}")
            seen[e[4]] = e[0]

        random.Random(args.seed).shuffle(entries)

        tasks = []
        for i in range(0, len(entries), args.size):
            part = entries[i:i + args.size]
            path = os.path.join(args.out, f"{split}{i // args.size:02d}-{len(part)}.tfrec")
            tasks.append((path, part))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            written = sum(ex.map(_write_shard, tasks))

        for path, part in tasks:
            base = os.path.basename(path)
            for tag, _, _, _, name, lab in part:
                manifest.write(f"{split},{base},{name},{lab},{tag}\n")

        pos = sum(e[5] for e in entries)
        grand += written
        print(f"{split:6s} " + " ".join(f"{per_src[t]:9d}" for t, _ in sources) +
              f" {written:9d} {100.0*pos/written:5.1f}% {len(tasks):7d}")

    manifest.close()
    print(f"total: {grand} imagens em {args.out}")
    print(f"provenance.csv escrito (mapeia cada imagem ao dataset de origem)")


if __name__ == "__main__":
    main()
