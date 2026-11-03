#!/usr/bin/env python3
import argparse
import collections
import csv
import os
import random
import sys

SPLITS = ("train", "val", "test")


def load_manifest(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            lab = int(r["retinopatia"])
            if lab not in (0, 1):
                sys.exit(f"{path}: rotulo {lab} fora de {{0,1}} em {r['imagem']}")
            rows.append((r["imagem"].strip(), lab, r["grupo"].strip()))
    if not rows:
        sys.exit(f"{path}: vazio")
    return rows


def split_groups(rows, ratios, seed):
    by_group = collections.defaultdict(list)
    for name, lab, grp in rows:
        by_group[grp].append((name, lab))

    strata = collections.defaultdict(list)
    for grp, imgs in by_group.items():
        strata[max(lab for _, lab in imgs)].append(grp)

    total = len(rows)
    quota = {s: total * r for s, r in zip(SPLITS, ratios)}
    got = {s: 0 for s in SPLITS}
    out = {s: [] for s in SPLITS}

    rng = random.Random(seed)
    for lab in sorted(strata):
        groups = sorted(strata[lab])
        rng.shuffle(groups)
        groups.sort(key=lambda g: -len(by_group[g]))
        for grp in groups:
            s = max(SPLITS, key=lambda k: (quota[k] - got[k]) / quota[k] if quota[k] else -1)
            out[s].extend(by_group[grp])
            got[s] += len(by_group[grp])
    return out, by_group


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="dir onde gravar labels_<split>.csv")
    ap.add_argument("--ratios", default="0.591,0.104,0.304",
                    help="train,val,test (default = proporcao do all-tfrec-v2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or any(r <= 0 for r in ratios):
        sys.exit(f"--ratios precisa de 3 valores positivos (recebi {ratios})")
    total_r = sum(ratios)
    if abs(total_r - 1.0) > 0.02:
        sys.exit(f"--ratios soma {total_r:.3f}, esperado ~1.0")
    ratios = tuple(r / total_r for r in ratios)

    rows = load_manifest(args.manifest)
    out, by_group = split_groups(rows, ratios, args.seed)

    os.makedirs(args.out, exist_ok=True)
    total = len(rows)
    print(f"{total} imagens, {len(by_group)} grupos, seed={args.seed}")
    for s in SPLITS:
        imgs = out[s]
        pos = sum(l for _, l in imgs)
        path = os.path.join(args.out, f"labels_{s}.csv")
        with open(path, "w") as w:
            w.write("imagem,retinopatia\n")
            for name, lab in sorted(imgs):
                w.write(f"{name},{lab}\n")
        print(f"  {s:5s}: {len(imgs):6d} ({100.0*len(imgs)/total:4.1f}%)  "
              f"prev={100.0*pos/len(imgs):4.1f}%  -> {os.path.basename(path)}")

    where = {}
    for s in SPLITS:
        for name, _ in out[s]:
            where[name] = s
    bad = 0
    for grp, imgs in by_group.items():
        splits_of_group = {where[n] for n, _ in imgs}
        if len(splits_of_group) > 1:
            bad += 1
    if bad:
        sys.exit(f"ERRO: {bad} grupos ficaram em mais de um split")
    print(f"  validacao: 0 grupos divididos entre splits (de {len(by_group)})")


if __name__ == "__main__":
    main()
