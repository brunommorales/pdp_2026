#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys

import numpy as np

STEADY_FROM = 5
ES_PATIENCE = 20

COLUMNS = ["approach", "run", "test_auc", "test_precision", "test_f1", "test_sens",
           "test_spec", "test_spec_at_sens95", "throughput_img_s", "gpu_mem_peak_mb",
           "avg_power_w", "avg_power_derived_w", "train_compute_s", "steady_epoch_time_s",
           "best_auc_epoch", "total_time_s", "total_energy_kj", "gpu_util_pct",
           "mem_util_pct", "busy_time_s", "energy_to_best_kj", "energy_wasted_kj",
           "energy_wasted_pct", "es_epoch", "es_best_epoch", "energy_es_kj",
           "energy_saved_kj", "energy_saved_pct", "es_auc_penalty", "val_energy_kj",
           "energy_job_kj"]


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in (None, "", "nan") else float("nan")


def spec_at_sens95(path):
    best = float("nan")
    for r in csv.DictReader(open(path)):
        s, sp = float(r["sens"]), float(r["spec"])
        if s >= 0.95 and (np.isnan(best) or sp > best):
            best = sp
    return best


def derive(run_dir, approach, run):
    thr = glob.glob(os.path.join(run_dir, "*-thresholds.csv"))
    hist = [p for p in glob.glob(os.path.join(run_dir, "*.csv"))
            if not p.endswith("-thresholds.csv")]
    if len(hist) != 1:
        raise RuntimeError(f"{run_dir}: esperava 1 CSV de historico, achei {hist}")

    rows = list(csv.DictReader(open(hist[0])))
    ep = [r for r in rows if r["epoch"].isdigit() and r.get("stage") != "final_test"]
    fin = [r for r in rows if r.get("test_auc") not in (None, "", "nan")][-1]
    st = [r for r in ep if int(r["epoch"]) >= STEADY_FROM]

    tr_e = np.array([_f(r, "train_energy_j") for r in ep])
    va_e = np.array([_f(r, "val_energy_j") for r in ep])
    vauc = np.array([_f(r, "val_auc") for r in ep])

    mean = lambda k: float(np.nanmean([_f(r, k) for r in st]))
    best_i = int(np.nanargmax(vauc))
    tot_e = float(np.nansum(tr_e))
    tot_t = _f(fin, "total_train_time_s")

    b, es_epoch = 0, len(vauc) - 1
    for i in range(1, len(vauc)):
        if vauc[i] > vauc[b]:
            b = i
        elif i - b >= ES_PATIENCE:
            es_epoch = i
            break
    e_es = float(np.nansum(tr_e[:es_epoch + 1]))

    return {
        "approach": approach, "run": run,
        "test_auc": _f(fin, "test_auc"), "test_precision": _f(fin, "test_precision"),
        "test_f1": _f(fin, "test_f1"), "test_sens": _f(fin, "test_sens"),
        "test_spec": _f(fin, "test_spec"),
        "test_spec_at_sens95": spec_at_sens95(thr[0]) if thr else float("nan"),
        "throughput_img_s": mean("train_throughput_img_s"),
        "gpu_mem_peak_mb": mean("train_gpu_mem_peak_mb"),
        "avg_power_w": mean("train_avg_power_w"),
        "avg_power_derived_w": tot_e / tot_t,
        "train_compute_s": float(np.nansum([_f(r, "train_elapsed_s") for r in ep])),
        "steady_epoch_time_s": float(np.nanmedian([_f(r, "train_elapsed_s") for r in st])),
        "best_auc_epoch": best_i,
        "total_time_s": tot_t,
        "total_energy_kj": tot_e / 1000.0,
        "gpu_util_pct": mean("train_gpu_util_pct"),
        "mem_util_pct": mean("train_mem_util_pct"),
        "busy_time_s": float(np.nansum([_f(r, "train_busy_time_s") for r in ep])),
        "energy_to_best_kj": float(np.nansum(tr_e[:best_i + 1])) / 1000.0,
        "energy_wasted_kj": (tot_e - float(np.nansum(tr_e[:best_i + 1]))) / 1000.0,
        "energy_wasted_pct": 100.0 * (tot_e - float(np.nansum(tr_e[:best_i + 1]))) / tot_e,
        "es_epoch": es_epoch, "es_best_epoch": b,
        "energy_es_kj": e_es / 1000.0,
        "energy_saved_kj": (tot_e - e_es) / 1000.0,
        "energy_saved_pct": 100.0 * (tot_e - e_es) / tot_e,
        "es_auc_penalty": float(vauc[best_i] - vauc[b]),
        "val_energy_kj": float(np.nansum(va_e)) / 1000.0,
        "energy_job_kj": float(np.nansum(tr_e) + np.nansum(va_e)) / 1000.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, metavar="APPROACH=DIR",
                    help="pode repetir; DIR contem run_0..run_9")
    ap.add_argument("--out", default="", help="CSV de saida (vazio = so imprime)")
    ap.add_argument("--validate", default="",
                    help="CSV de referencia para conferir as formulas antes de gravar")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    srcs = []
    for s in args.src:
        a, d = s.split("=", 1)
        srcs.append((a, d))

    out = []
    for a, d in srcs:
        for i in range(args.runs):
            rd = os.path.join(d, f"run_{i}")
            if not os.path.isdir(rd):
                sys.exit(f"faltando: {rd}")
            out.append(derive(rd, a, str(i)))

    if args.validate:
        ref = {(r["approach"], r["run"]): r for r in csv.DictReader(open(args.validate))}
        worst, checked = {}, 0
        for g in out:
            w = ref.get((g["approach"], g["run"]))
            if not w:
                continue
            checked += 1
            for c in COLUMNS[2:]:
                v = w.get(c, "")
                if v in (None, "", "nan"):
                    continue
                v = float(v)
                abs_e = abs(g[c] - v)
                rel_e = abs_e / (abs(v) if abs(v) > 1e-9 else 1.0)
                worst[c] = max(worst.get(c, 0.0), min(abs_e, rel_e))
        print(f"validacao contra {args.validate}: {checked} runs em comum")
        bad = [c for c, e in worst.items() if e > 1e-5]
        for c in COLUMNS[2:]:
            if c in worst:
                print(f"  {c:24s} erro rel max = {worst[c]:.2e}"
                      f"{'   <-- DIVERGE' if worst[c] > 1e-5 else ''}")
        if bad:
            sys.exit(f"\n{len(bad)} colunas divergem: {bad}")
        print("  todas as colunas conferem")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in out:
                w.writerow(r)
        print(f"gravado: {args.out} ({len(out)} linhas)")


if __name__ == "__main__":
    main()
