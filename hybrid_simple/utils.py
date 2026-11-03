import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_shared.training_utils import count_parameters


def get_backbone_feature_dim(backbone_name: str, img_size: int) -> int:
    feature_dims = {
        "inception_v3": 2048,
        "inception_resnet_v2": 1536,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "xception": 2048,
        "vgg16": 512,
        "vgg19": 512,
    }

    return feature_dims.get(backbone_name, 2048)


def modify_backbone_for_feature_extraction(backbone: nn.Module, backbone_name: str) -> nn.Module:
    if hasattr(backbone, 'fc'):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, 'classifier'):
        backbone.classifier = nn.Identity()
    elif hasattr(backbone, '_fc'):
        backbone._fc = nn.Identity()

    return backbone


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    return trainable, total


def model_summary(model: nn.Module, model_name: str = "Hybrid Simple") -> str:
    trainable, total = count_model_parameters(model)

    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  MODEL SUMMARY: {model_name:<40}║
╚════════════════════════════════════════════════════════════════╝

Parâmetros:
  Treináveis:     {trainable:>15,}
  Totais:         {total:>15,}
  Taxa treino:    {trainable/total*100:>14.1f}%

Estrutura:
{model}
"""
    return summary


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_gpu_memory_info() -> None:
    if not torch.cuda.is_available():
        print("GPU não disponível")
        return

    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    try:
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    except:
        pass
