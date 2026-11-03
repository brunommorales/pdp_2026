#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import sys

HEADER = 8 + 4
FOOTER = 4
FRAME = HEADER + FOOTER


def record_offsets(tfrec_path):
    offsets = []
    size = os.path.getsize(tfrec_path)
    with open(tfrec_path, "rb") as fh:
        pos = 0
        while pos < size:
            fh.seek(pos)
            head = fh.read(8)
            if len(head) < 8:
                raise ValueError(
                    f"{tfrec_path}: cabecalho truncado em offset {pos} "
                    f"(lidos {len(head)} de 8 bytes)"
                )
            (length,) = struct.unpack("<Q", head)
            nxt = pos + length + FRAME
            if nxt > size:
                raise ValueError(
                    f"{tfrec_path}: registro em {pos} declara {length} bytes, "
                    f"o que ultrapassa o fim do arquivo ({nxt} > {size})"
                )
            offsets.append(pos)
            pos = nxt
    return offsets


def expected_count(tfrec_path):
    base = os.path.basename(tfrec_path)
    stem = base[:-len(".tfrec")] if base.endswith(".tfrec") else base
    if "-" not in stem:
        return None
    tail = stem.rsplit("-", 1)[1]
    return int(tail) if tail.isdigit() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="diretorio com os *.tfrec")
    ap.add_argument("--glob", default="*.tfrec", help="padrao dos shards")
    ap.add_argument("--check", action="store_true",
                    help="nao escreve; so confere os .idx que ja existem")
    ap.add_argument("--force", action="store_true",
                    help="reescreve .idx mesmo que ja exista e esteja correto")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        sys.exit(f"nenhum shard casou com {args.glob} em {args.dir}")

    total, written, ok, bad = 0, 0, 0, 0
    for tfrec in files:
        idx_path = tfrec + ".idx"
        offsets = record_offsets(tfrec)
        total += len(offsets)

        want = expected_count(tfrec)
        if want is not None and want != len(offsets):
            print(f"  ! {os.path.basename(tfrec)}: nome diz {want} registros, "
                  f"o arquivo tem {len(offsets)}")
            bad += 1

        if args.check:
            if not os.path.exists(idx_path):
                print(f"  ! {os.path.basename(idx_path)}: AUSENTE")
                bad += 1
                continue
            have = [int(l.split()[0]) for l in open(idx_path) if l.strip()]
            if have == offsets:
                ok += 1
            else:
                print(f"  ! {os.path.basename(idx_path)}: DIVERGE "
                      f"({len(have)} entradas no .idx vs {len(offsets)} no .tfrec)")
                bad += 1
            continue

        if os.path.exists(idx_path) and not args.force:
            have = [int(l.split()[0]) for l in open(idx_path) if l.strip()]
            if have == offsets:
                ok += 1
                continue

        tmp = idx_path + ".tmp"
        with open(tmp, "w") as w:
            w.write("".join(f"{o}\n" for o in offsets))
        os.replace(tmp, idx_path)
        written += 1

    verbo = "conferidos" if args.check else "gravados"
    print(f"{len(files)} shards, {total} registros | {verbo}={written} "
          f"ja_corretos={ok} problemas={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
