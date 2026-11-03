#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_val_split import TFRecordWriter, build_example
from make_tfrec_idx import record_offsets


def read_labels(csv_path):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            name = r["imagem"].strip()
            lab = int(r["retinopatia"])
            if lab not in (0, 1):
                raise ValueError(f"{csv_path}: rotulo {lab} fora de {{0,1}} em {name}")
            rows.append((name, lab))
    if not rows:
        raise ValueError(f"{csv_path}: nenhuma linha")
    return rows


def _write_one_shard(task):
    images_dir, path, rows = task
    pos = 0
    w = TFRecordWriter(path)
    try:
        for name, lab in rows:
            with open(os.path.join(images_dir, name), "rb") as fh:
                payload = fh.read()
            w.write(build_example({
                "imagem": ("bytes", payload),
                "image_name": ("bytes", name.encode()),
                "retinopatia": ("int", lab),
            }))
            pos += lab
    finally:
        w.close()

    offsets = record_offsets(path)
    if len(offsets) != len(rows):
        raise RuntimeError(
            f"{path}: gravei {len(rows)} registros mas o indice achou {len(offsets)}"
        )
    tmp = path + ".idx.tmp"
    with open(tmp, "w") as idx:
        idx.write("".join(f"{o}\n" for o in offsets))
    os.replace(tmp, path + ".idx")
    return len(rows), pos


def write_split(images_dir, out_dir, split, rows, shard_size, workers):
    tasks = []
    for s in range(0, len(rows), shard_size):
        part = rows[s:s + shard_size]
        path = os.path.join(out_dir, f"{split}{s // shard_size:02d}-{len(part)}.tfrec")
        tasks.append((images_dir, path, part))

    written = pos = 0
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n, p in ex.map(_write_one_shard, tasks):
                written += n
                pos += p
    else:
        for t in tasks:
            n, p = _write_one_shard(t)
            written += n
            pos += p

    return len(tasks), written, pos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="dir com as imagens pre-processadas")
    ap.add_argument("--out", required=True, help="dir de saida dos .tfrec")
    ap.add_argument("--labels-dir", default="",
                    help="dir dos labels_<split>.csv (default: igual a --images)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--size", type=int, default=100,
                    help="imagens por shard (all-tfrec-v2 usa 100)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="processos para gravar shards em paralelo")
    args = ap.parse_args()

    labels_dir = args.labels_dir or args.images
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    plan = {}
    for split in splits:
        csv_path = os.path.join(labels_dir, f"labels_{split}.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"faltando: {csv_path}")
        rows = read_labels(csv_path)
        missing = [n for n, _ in rows if not os.path.exists(os.path.join(args.images, n))]
        if missing:
            sys.exit(f"{split}: {len(missing)} imagens ausentes, ex: {missing[:3]}")
        plan[split] = rows

    seen = {}
    for split, rows in plan.items():
        for name, _ in rows:
            if name in seen:
                sys.exit(f"VAZAMENTO: {name} esta em '{seen[name]}' e em '{split}'")
            seen[name] = split

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for split in splits:
        rows = plan[split]
        n_sh, n_img, n_pos = write_split(args.images, args.out, split, rows,
                                         args.size, args.workers)
        total += n_img
        print(f"  {split:5s}: {n_sh:3d} shards, {n_img:6d} imagens, "
              f"prev={100.0 * n_pos / n_img:.1f}%")
    print(f"total: {total} imagens em {args.out}")


if __name__ == "__main__":
    main()
