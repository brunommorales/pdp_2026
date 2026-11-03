from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from nvidia.dali import fn
from nvidia.dali import types as dali_types
from nvidia.dali import tfrecord as dali_tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def

TFRECORD_FEATURES = {
    "imagem": dali_tfrec.FixedLenFeature((), dali_tfrec.string, ""),
    "retinopatia": dali_tfrec.FixedLenFeature([1], dali_tfrec.int64, 0),
}


def _as_string_list(paths: Iterable[Path | str]) -> List[str]:
    return [str(Path(p)) for p in paths]


def _clamp_images(data_node, *, min_value: float, max_value: float):
    if hasattr(fn, "clamp"):
        return fn.clamp(data_node, min=min_value, max=max_value)

    return data_node


@pipeline_def
def _retina_tfrecord_pipeline(
    tfrec_files: Sequence[str],
    idx_files: Sequence[str],
    *,
    shard_id: int,
    num_shards: int,
    reader_seed: int,
    image_size: int,
    enable_augment: bool,
    output_layout: str = "NHWC",
    fundus_crop_ratio: float = 1.0,
    random_shuffle: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    assert output_layout in ("NHWC", "NCHW")

    reader = fn.readers.tfrecord(
        path=tfrec_files,
        index_path=idx_files,
        features=TFRECORD_FEATURES,
        random_shuffle=random_shuffle,
        initial_fill=1024,
        read_ahead=True,
        prefetch_queue_depth=2,
        name="Reader",
        seed=reader_seed,
        skip_cached_images=False,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    images = fn.decoders.image(
        reader["imagem"],
        device="mixed",
        output_type=dali_types.RGB,
    )

    if fundus_crop_ratio < 0.999:
        images = fn.crop(
            images,
            crop=[fundus_crop_ratio, fundus_crop_ratio],
            crop_pos_x=0.5,
            crop_pos_y=0.5,
        )

    images = fn.resize(
        images,
        resize_x=image_size,
        resize_y=image_size,
        interp_type=dali_types.INTERP_TRIANGULAR,
    )
    images = fn.cast(images, dtype=dali_types.FLOAT)

    if enable_augment:
        mirror_h = fn.random.coin_flip(probability=0.5)
        mirror_v = fn.random.coin_flip(probability=0.1)
        images = fn.flip(images, horizontal=mirror_h, vertical=mirror_v)

        rotate_cond = fn.random.coin_flip(probability=0.25)
        rotate_angle = fn.random.uniform(range=(-10.0, 10.0)) * rotate_cond
        images = fn.rotate(
            images,
            angle=rotate_angle,
            keep_size=True,
            fill_value=128.0,
        )

        contrast_random = fn.random.uniform(range=(0.9, 1.1))
        contrast_cond = fn.random.coin_flip(probability=0.3)
        contrast_factor = 1.0 + (contrast_random - 1.0) * contrast_cond

        brightness_random = fn.random.uniform(range=(0.9, 1.1))
        brightness_cond = fn.random.coin_flip(probability=0.3)
        brightness_factor = 1.0 + (brightness_random - 1.0) * brightness_cond

        images = fn.brightness_contrast(
            images,
            contrast=contrast_factor,
            brightness=brightness_factor,
        )

        saturation_random = fn.random.uniform(range=(0.9, 1.1))
        saturation_cond = fn.random.coin_flip(probability=0.2)
        saturation_factor = 1.0 + (saturation_random - 1.0) * saturation_cond
        images = fn.color_twist(images, saturation=saturation_factor)

        images = _clamp_images(images, min_value=0.0, max_value=255.0)

    dali_output_layout = "HWC" if output_layout == "NHWC" else "CHW"

    mean = [127.5, 127.5, 127.5] if mean is None else mean
    std = [127.5, 127.5, 127.5] if std is None else std

    images = fn.crop_mirror_normalize(
        images,
        dtype=dali_types.FLOAT,
        output_layout=dali_output_layout,
        mean=mean,
        std=std,
    )

    labels = reader["retinopatia"]
    labels = fn.cast(labels, dtype=dali_types.FLOAT)
    labels = labels.gpu()

    return images, labels


def build_dali_pipeline(
    tfrec_files: Sequence[Path | str],
    idx_files: Sequence[Path | str],
    *,
    batch_size: int,
    image_size: int = 299,
    num_threads: int = 4,
    device_id: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
    seed: int = 2024,
    enable_augment: bool = False,
    output_layout: str = "NHWC",
    fundus_crop_ratio: float = 1.0,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    random_shuffle: bool = True,
):
    tfrec_files = _as_string_list(tfrec_files)
    idx_files = _as_string_list(idx_files)

    pipe = _retina_tfrecord_pipeline(
        tfrec_files=tfrec_files,
        idx_files=idx_files,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=seed,
        reader_seed=seed,
        image_size=image_size,
        enable_augment=enable_augment,
        output_layout=output_layout,
        fundus_crop_ratio=fundus_crop_ratio,
        random_shuffle=random_shuffle,
        mean=mean,
        std=std,
    )
    return pipe


def create_dali_iterator(
    pipelines: Sequence,
    *,
    auto_reset: bool = True,
    last_batch_policy: LastBatchPolicy = LastBatchPolicy.PARTIAL,
):
    return DALIGenericIterator(
        pipelines,
        output_map=["image", "label"],
        reader_name="Reader",
        auto_reset=auto_reset,
        last_batch_policy=last_batch_policy,
        prepare_first_batch=True,
    )


def _discover_tfrec_pairs(tfrec_dir: Path) -> Tuple[List[Path], List[Path]]:
    tfrecs = sorted(tfrec_dir.glob("train*.tfrec"))
    idxs = [p.with_suffix(".tfrec.idx") for p in tfrecs]
    if not tfrecs:
        raise FileNotFoundError(f"Nenhum train*.tfrec em: {tfrec_dir}")
    for p in idxs:
        if not p.exists():
            raise FileNotFoundError(f"Idx não encontrado: {p}")
    return tfrecs, idxs


def demo_cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test do DALI TFRecord → PyTorch.")
    parser.add_argument("--tfrec_dir", type=str, default="/home/users/bmmorales/projects/hcpa/data/all-tfrec", help="Pasta com train*.tfrec")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--layout", type=str, default="NHWC", choices=["NHWC", "NCHW"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Imprime shapes/dtypes/devices")
    args = parser.parse_args()

    tfrec_dir = Path(args.tfrec_dir)
    tfrec_files, idx_files = _discover_tfrec_pairs(tfrec_dir)

    pipe = build_dali_pipeline(
        tfrec_files=tfrec_files,
        idx_files=idx_files,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_threads=args.threads,
        device_id=args.device_id,
        shard_id=0,
        num_shards=1,
        seed=2024,
        enable_augment=args.augment,
        output_layout=args.layout,
    )
    pipe.build()
    iterator = create_dali_iterator([pipe])

    batch = next(iter(iterator))
    images = batch[0]["image"]
    labels = batch[0]["label"]

    print("Images:", tuple(images.shape), images.dtype, images.device)
    print("Labels:", tuple(labels.shape), labels.dtype, labels.device)
    try:
        import torch
        print("Example label values:", torch.flatten(labels[:8]).detach().cpu().numpy())
    except Exception:
        pass


if __name__ == "__main__":
    demo_cli()
