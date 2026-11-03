import math
import os

MODOS = ("none", "sqrt", "linear")


def world_size() -> int:
    try:
        n = int(os.environ.get("HCPA_WORLD_SIZE", "1"))
    except ValueError:
        return 1
    return n if n > 0 else 1


def fator_escala() -> float:
    modo = os.environ.get("HCPA_LR_SCALE", "none").strip().lower()
    n = world_size()

    if modo not in MODOS:
        raise ValueError(
            f"HCPA_LR_SCALE={modo!r} invalido; use um de {MODOS}"
        )
    if modo == "none" or n <= 1:
        return 1.0
    return math.sqrt(n) if modo == "sqrt" else float(n)


def escalar_lr(lr: float, rotulo: str = "lrate") -> float:
    fator = fator_escala()
    novo = lr * fator

    modo = os.environ.get("HCPA_LR_SCALE", "none").strip().lower()
    print(
        f"[LR] {rotulo}: {lr:.3e} -> {novo:.3e} "
        f"(modo={modo}, world_size={world_size()}, fator={fator:.4f})",
        flush=True,
    )
    return novo
