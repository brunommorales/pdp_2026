#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import re
import sys
import time

HCPA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HCPA, "tensorflow_opt"))
from lib.preprocess import resize_and_center_fundus

DIAMETER = 299
BIN = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
PATIENT_RE = re.compile(r"^(\d+)_(left|right)$")

OUT = None


def work(task):
    path, fn = task
    try:
        if resize_and_center_fundus(save_path=OUT, image_paths=[path],
                                    diameter=DIAMETER, verbosity=0) != 1:
            return fn
    except Exception as e:
        sys.stderr.write(f"\n{fn}: {e}\n")
        return fn
    return None


def collect(raw):
    root = os.path.join(raw, "dr_unified_v2", "dr_unified_v2")
    if not os.path.isdir(root):
        sys.exit(f"nao encontrei {root}")
    out, skipped_gf, seen = [], 0, {}
    for split in sorted(os.listdir(root)):
        sd = os.path.join(root, split)
        if not os.path.isdir(sd):
            continue
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if not os.path.isdir(cd) or not cls.isdigit():
                continue
            lab = BIN[int(cls)]
            for fn in sorted(os.listdir(cd)):
                if not fn.lower().endswith(".jpg"):
                    continue
                stem = fn[:-4]
                if stem.endswith("-GF") or stem.startswith("GF-"):
                    skipped_gf += 1
                    continue
                if stem in seen:
                    sys.stderr.write(f"aviso: nome repetido {fn} "
                                     f"({seen[stem]} e {split}/{cls}); mantida a 1a\n")
                    continue
                seen[stem] = f"{split}/{cls}"
                out.append((os.path.join(cd, fn), fn, lab))
    print(f"- coletadas {len(out)} imagens | descartadas copias GF: {skipped_gf}")

    stems = {os.path.splitext(fn)[0] for _, fn, _ in out}
    dup = []
    for s in stems:
        for core in (s[3:] if s.startswith("GF-") else None,
                     s[:-3] if s.endswith("-GF") else None):
            if core and core in stems:
                dup.append(s)
    if dup:
        sys.exit(f"ainda ha {len(dup)} nomes que sao variantes de outros, "
                 f"ex: {sorted(dup)[:5]}")
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="dir onde o zip foi extraido")
    ap.add_argument("--out", required=True, help="dir de saida das imagens 299px")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="so regera o manifest.csv, reusando as imagens ja processadas")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    OUT = os.path.abspath(args.out)
    os.makedirs(OUT, exist_ok=True)
    items = collect(os.path.abspath(args.raw))

    failed = set()
    if args.skip_preprocess:
        have = set(os.listdir(OUT))
        missing = [fn for _, fn, _ in items if fn not in have]
        if len(missing) > 0.01 * len(items):
            sys.exit(f"--skip-preprocess: faltam {len(missing)} de {len(items)} "
                     f"imagens ({100.0*len(missing)/len(items):.1f}%), ex: {missing[:3]}. "
                     f"Diretorio errado? Rode sem a flag.")
        failed.update(missing)
        print(f"- preprocess pulado: {len(items)-len(missing)} imagens presentes, "
              f"{len(missing)} ausentes (falhas do preprocess original)")
    else:
        t0, n = time.time(), len(items)
        tasks = [(p, fn) for p, fn, _ in items]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(work, tasks, chunksize=16), 1):
                if r is not None:
                    failed.add(r)
                if i % 500 == 0 or i == n:
                    sys.stdout.write(f"\r- preprocessing {i}/{n} falhas={len(failed)}")
                    sys.stdout.flush()
        print(f"\n- preprocess em {round(time.time()-t0,1)}s, falhas={len(failed)}")

    path = os.path.join(OUT, "manifest.csv")
    kept = npat = 0
    with open(path, "w") as w:
        w.write("imagem,retinopatia,grupo\n")
        for _, fn, lab in items:
            if fn in failed or not os.path.exists(os.path.join(OUT, fn)):
                continue
            stem = fn[:-4]
            m = PATIENT_RE.match(stem)
            grupo = m.group(1) if m else stem
            if m:
                npat += 1
            w.write(f"{fn},{lab},{grupo}\n")
            kept += 1
    print(f"- manifest: {kept} linhas -> {path}")
    print(f"  {npat} com id de paciente (EyePACS), {kept-npat} sem (APTOS/Messidor)")


if __name__ == "__main__":
    main()
