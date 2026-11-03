#!/usr/bin/env python3
import argparse
import os
import re
import struct
import glob
import random
import collections

def _make_tables():
    poly = 0x82F63B78
    t0 = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ poly if (c & 1) else (c >> 1)
        t0.append(c & 0xFFFFFFFF)
    tabs = [t0]
    for _ in range(7):
        prev = tabs[-1]
        cur = []
        for n in range(256):
            c = prev[n]
            cur.append(t0[c & 0xFF] ^ (c >> 8))
        tabs.append(cur)
    return tabs


_CRC_TABS = _make_tables()
_CRC_TABLE = _CRC_TABS[0]


def crc32c(data: bytes, crc=0):
    t0, t1, t2, t3, t4, t5, t6, t7 = _CRC_TABS
    crc ^= 0xFFFFFFFF
    n = len(data)
    i = 0
    limit = n - (n % 8)
    while i < limit:
        crc ^= (data[i] | data[i + 1] << 8 | data[i + 2] << 16 | data[i + 3] << 24)
        b4, b5, b6, b7 = data[i + 4], data[i + 5], data[i + 6], data[i + 7]
        crc = (t7[crc & 0xFF] ^ t6[(crc >> 8) & 0xFF] ^ t5[(crc >> 16) & 0xFF]
               ^ t4[(crc >> 24) & 0xFF] ^ t3[b4] ^ t2[b5] ^ t1[b6] ^ t0[b7])
        i += 8
    while i < n:
        crc = (crc >> 8) ^ t0[(crc ^ data[i]) & 0xFF]
        i += 1
    return crc ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_records(path):
    for _blk, payload in read_raw_records(path):
        yield payload


def read_raw_records(path):
    with open(path, "rb") as fh:
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            ln = struct.unpack("<Q", hdr)[0]
            crc_len = fh.read(4)
            payload = fh.read(ln)
            crc_pl = fh.read(4)
            blk = hdr + crc_len + payload + crc_pl
            yield blk, payload


class TFRecordWriter:
    def __init__(self, path):
        self.fh = open(path, "wb")
        self.n = 0

    def write(self, payload: bytes):
        ln = struct.pack("<Q", len(payload))
        self.fh.write(ln)
        self.fh.write(struct.pack("<I", masked_crc(ln)))
        self.fh.write(payload)
        self.fh.write(struct.pack("<I", masked_crc(payload)))
        self.n += 1

    def close(self):
        self.fh.close()


def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def _fields(b):
    i = 0
    while i < len(b):
        k, i = _varint(b, i)
        fn, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, wt, b[i:i + ln]; i += ln
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        else:
            return


def parse_example(buf):
    out = {}
    for fn, _, v in _fields(buf):
        if fn != 1:
            continue
        for _, _, entry in _fields(v):
            key = val = None
            for fn3, wt3, v3 in _fields(entry):
                if fn3 == 1 and wt3 == 2:
                    key = v3.decode(errors="replace")
                elif fn3 == 2 and wt3 == 2:
                    for fn4, _, v4 in _fields(v3):
                        if fn4 == 1:
                            for _, _, s in _fields(v4):
                                val = ("bytes", s)
                        elif fn4 == 3:
                            for _, wt5, n in _fields(v4):
                                if wt5 == 0:
                                    val = ("int", n)
                                elif wt5 == 2:
                                    x, _ = _varint(n, 0); val = ("int", x)
            if key:
                out[key] = val
    return out


def _tag(fn, wt):
    return _enc_varint((fn << 3) | wt)


def _enc_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _len_delim(fn, payload):
    return _tag(fn, 2) + _enc_varint(len(payload)) + payload


def _bytes_feature(value: bytes):
    bl = _len_delim(1, value)
    return _len_delim(1, bl)


def _int_feature(value: int):
    il = _tag(1, 0) + _enc_varint(value)
    return _len_delim(3, il)


def build_example(features: dict):
    feats = b""
    for name, (kind, val) in features.items():
        f = _bytes_feature(val) if kind == "bytes" else _int_feature(val)
        entry = _len_delim(1, name.encode()) + _len_delim(2, f)
        feats += _len_delim(1, entry)
    return _len_delim(1, feats)


def patient_and_side(name):
    b = name[:-4] if name.lower().endswith((".jpg", ".png")) else name
    side = -1
    m = re.search(r"-(\d+)$", b)
    if m:
        try:
            side = int(m.group(1)) & 1
        except ValueError:
            side = -1
    core = re.sub(r"-\d+$", "", b)
    if re.fullmatch(r"\d{18,25}", core):
        return ("num:" + core[:18], side)
    if core.startswith("1."):
        return ("dcm:" + ".".join(core.split(".")[:9]), side)
    return ("img:" + core, side)


def stratified_group_split(examples, val_frac, seed):
    by_pat = collections.defaultdict(list)
    for e in examples:
        by_pat[e["patient"]].append(e["label"])
    pat = {p: (len(ls), sum(ls)) for p, ls in by_pat.items()}

    tot_img = sum(n for n, _ in pat.values())
    tot_pos = sum(k for _, k in pat.values())
    tgt_img = val_frac * tot_img
    tgt_pos = val_frac * tot_pos

    rng = random.Random(seed)
    order = list(pat)
    rng.shuffle(order)

    val_pat = set()
    v_img = v_pos = 0
    remaining = list(order)
    while v_img < tgt_img and remaining:
        need_img = max(tgt_img - v_img, 1e-9)
        need_pos = max(tgt_pos - v_pos, 1e-9)
        best, best_score = None, None
        for p in remaining:
            n, k = pat[p]
            over_img = max(0, (v_img + n) - tgt_img) / need_img
            over_pos = max(0, (v_pos + k) - tgt_pos) / need_pos
            fill = (n / need_img) + (k / need_pos)
            score = fill - 2.0 * (over_img + over_pos)
            if best_score is None or score > best_score:
                best, best_score = p, score
        val_pat.add(best)
        n, k = pat[best]
        v_img += n; v_pos += k
        remaining.remove(best)

    train_pat = set(pat) - val_pat
    return train_pat, val_pat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/all-tfrec")
    ap.add_argument("--dst", default="data/all-tfrec-v2")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-file", type=int, default=100)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)

    print("lendo train*.tfrec ...")
    train_examples = []
    for f in sorted(glob.glob(os.path.join(a.src, "train*.tfrec"))):
        for blk, payload in read_raw_records(f):
            ex = parse_example(payload)
            name = ex["image_name"][1].decode() if "image_name" in ex else None
            label = ex["retinopatia"][1] if "retinopatia" in ex else None
            if name is None or label is None:
                raise SystemExit(f"Example sem image_name/retinopatia em {f}")
            pat, side = patient_and_side(name)
            train_examples.append(dict(blk=blk, name=name, label=int(label),
                                       patient=pat, side=side))
    print(f"  {len(train_examples)} exemplos de treino")

    train_pat, val_pat = stratified_group_split(train_examples, a.val_frac, a.seed)
    assert train_pat.isdisjoint(val_pat), "vazamento de paciente train<->val"

    tr = [e for e in train_examples if e["patient"] in train_pat]
    va = [e for e in train_examples if e["patient"] in val_pat]

    def prev(xs):
        return sum(e["label"] for e in xs) / len(xs) if xs else 0.0
    print(f"  train': {len(tr)} imgs, {len(train_pat)} pacientes, prevalencia {prev(tr):.3f}")
    print(f"  val   : {len(va)} imgs, {len(val_pat)} pacientes, prevalencia {prev(va):.3f}")

    def write_split(examples, prefix):
        n = idx = 0
        chunk = a.per_file
        for start in range(0, len(examples), chunk):
            part = examples[start:start + chunk]
            path = os.path.join(a.dst, f"{prefix}{idx:02d}-{len(part)}.tfrec")
            with open(path, "wb") as fh:
                for e in part:
                    fh.write(e["blk"])
                    n += 1
            idx += 1
        return n

    print("regravando (movendo blocos brutos) ...")
    ntr = write_split(tr, "train")
    nva = write_split(va, "val")

    import shutil
    ntest = 0
    for f in sorted(glob.glob(os.path.join(a.src, "test*.tfrec"))):
        shutil.copy2(f, os.path.join(a.dst, os.path.basename(f)))
        ntest += sum(1 for _ in read_records(f))
    print(f"  test copiado intacto: {ntest} imgs")

    print("\nvalidando ...")
    back = collections.Counter()
    img_ok = True
    orig_imgs = {e["name"]: parse_example(e["blk"][12:12 + struct.unpack("<Q", e["blk"][:8])[0]])["imagem"][1]
                 for e in train_examples[:200]}
    checked = 0
    for split in ("train", "val"):
        for f in sorted(glob.glob(os.path.join(a.dst, f"{split}*.tfrec"))):
            for payload in read_records(f):
                ex = parse_example(payload)
                back[split] += 1
                nm = ex["image_name"][1].decode()
                if nm in orig_imgs and checked < 200:
                    if ex["imagem"][1] != orig_imgs[nm]:
                        img_ok = False
                    checked += 1
    print(f"  round-trip de bytes de imagem: {'OK' if img_ok else 'FALHOU'} ({checked} conferidos)")
    print(f"  contagem regravada: train={back['train']} val={back['val']} (esperado {ntr}/{nva})")
    assert back["train"] == ntr and back["val"] == nva, "contagem nao bate"

    def pats_of(split):
        s = set()
        for f in sorted(glob.glob(os.path.join(a.dst, f"{split}*.tfrec"))):
            for payload in read_records(f):
                nm = parse_example(payload)["image_name"][1].decode()
                s.add(patient_and_side(nm)[0])
        return s
    ptr, pva = pats_of("train"), pats_of("val")
    leak = ptr & pva
    print(f"  pacientes train'∩val: {len(leak)} (deve ser 0)")
    assert not leak, "VAZAMENTO de paciente train'<->val"

    print(f"\n=== OK. Split gravado em {a.dst}/ ===")
    print(f"  train*/val*/test*  |  imagens: {ntr}/{nva}/{ntest}")
    print(f"  val disjunto do train POR PACIENTE; test copiado intacto.")
    print(f"  (Examples preservados byte-a-byte do original; image_name mantido.)")


if __name__ == "__main__":
    main()
