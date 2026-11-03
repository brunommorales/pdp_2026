#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

NORMALIZE = "preprocess"
CROP_RATIO = 1.0
VALID_NORMALIZE = ("preprocess", "raw255", "unit")
assert NORMALIZE in VALID_NORMALIZE, f"normalize invalido: {NORMALIZE}"

HCPA = "/hcpa" if os.path.isdir("/hcpa") else os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

CFG = {
    "pytorch_opt":                dict(kind="cnn",    img=299),
    "pytorch_base":               dict(kind="cnn",    img=299),
    "retfound_green":             dict(kind="retf",   img=392),
    "hybrid_simple":              dict(kind="hybrid", img=299, layers=4, heads=4),
    "hybrid_token_reduction":     dict(kind="hybrid", img=299, layers=4, heads=4),
    "hybrid_token_reduction_opt": dict(kind="hybrid", img=299, layers=4, heads=4),
}


def load_module(path, name):
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def build(approach, device):
    cfg = CFG[approach]
    sys.path.insert(0, HCPA)
    if cfg["kind"] == "cnn":
        m = load_module(f"{HCPA}/{approach}/dr_hcpa_v2_2024.py", "dr")
        model = m.create_backbone("inception_v3", cfg["img"])
        if isinstance(model, (tuple, list)):
            model = model[0]
        return model.to(device), m, cfg
    if cfg["kind"] == "retf":
        sys.path.insert(0, f"{HCPA}/retfound_green")
        from model import create_retfound_green_model
        model = create_retfound_green_model(img_size=cfg["img"], checkpoint_path=None)
        m = load_module(f"{HCPA}/pytorch_opt/dr_hcpa_v2_2024.py", "dr")
        return model.to(device), m, cfg
    sys.path.insert(0, f"{HCPA}/{approach}")
    from hybrid_shared.data_loader_bridge import get_backbone
    from hybrid_simple.utils import modify_backbone_for_feature_extraction
    mod = load_module(f"{HCPA}/{approach}/model.py", f"{approach}_model")
    backbone = modify_backbone_for_feature_extraction(
        get_backbone(model_name="inception_v3", img_size=cfg["img"], pretrained=False),
        "inception_v3")
    factory = [getattr(mod, n) for n in dir(mod) if n.startswith("create_") and callable(getattr(mod, n))]
    if not factory:
        sys.exit(f"{approach}: nao achei create_* em model.py")
    model = factory[0](backbone=backbone, backbone_feature_dim=2048,
                       transformer_dim=256, num_transformer_layers=cfg["layers"],
                       num_heads=cfg["heads"])
    m = load_module(f"{HCPA}/pytorch_opt/dr_hcpa_v2_2024.py", "dr")
    return model.to(device), m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=sorted(CFG))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tfrec-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, m, cfg = build(args.approach, device)

    ck = torch.load(args.weights, map_location="cpu")
    sd = strip_prefixes(ck.get("state_dict", ck))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_tot = len(list(model.state_dict()))
    if len(missing) > 0.05 * n_tot:
        sys.exit(f"{args.approach}: {len(missing)}/{n_tot} chaves ausentes ao "
                 f"carregar {args.weights}. Arquitetura divergente. ex={missing[:5]}")
    print(f"[load] {args.approach}: faltando={len(missing)} inesperadas={len(unexpected)} "
          f"de {n_tot} chaves")

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    files = sorted(f for s in splits
                   for f in glob.glob(os.path.join(args.tfrec_dir, f"{s}*.tfrec")))
    if not files:
        sys.exit(f"nenhum {args.split}*.tfrec em {args.tfrec_dir}")
    if hasattr(m, "build_torch_loader"):
        loader = m.build_torch_loader(
            files, batch_size=args.batch_size, image_size=cfg["img"], world_size=1, rank=0,
            normalize_mode=NORMALIZE, augment=False, fundus_crop_ratio=CROP_RATIO,
            model_name="inception_v3", use_cuda=torch.cuda.is_available(), shuffle=False)[0]
    else:
        from pathlib import Path
        from torch.utils.data import DataLoader
        ds = m.RetinaTFRecord(
            [str(p) for p in files], [str(m.infer_idx(Path(p))) for p in files],
            img_size=cfg["img"], model_name="inception_v3", normalize_mode=NORMALIZE,
            augment=False, apply_normalization=True, fundus_crop_ratio=CROP_RATIO)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(8, os.cpu_count() or 4),
                            pin_memory=torch.cuda.is_available(), drop_last=False)

    model.eval()
    ys, ps = [], []
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0], batch[1]
            while isinstance(xb, (tuple, list)):
                xb = xb[0]
            while isinstance(yb, (tuple, list)):
                yb = yb[0]
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            ps.append(torch.sigmoid(out.float().flatten()).cpu().numpy())
            ys.append(yb.numpy().flatten())
    dt = time.perf_counter() - t0
    y = np.concatenate(ys); p = np.concatenate(ps)

    from sklearn.metrics import roc_auc_score, average_precision_score
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    res = dict(approach=args.approach, split=args.split, n=int(len(y)),
               prevalence=float(y.mean()), auc=float(roc_auc_score(y, p)),
               ap=float(average_precision_score(y, p)),
               sens=tp / (tp + fn) if tp + fn else float("nan"),
               spec=tn / (tn + fp) if tn + fp else float("nan"),
               precision=tp / (tp + fp) if tp + fp else float("nan"),
               tp=tp, tn=tn, fp=fp, fn=fn,
               seconds=round(dt, 1), img_s=round(len(y) / dt, 1))
    res["f1"] = (2 * res["precision"] * res["sens"] / (res["precision"] + res["sens"])
                 if res["precision"] + res["sens"] else float("nan"))
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
