# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import streaming
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm

from mosaicfm.data import DataCollator
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def setup() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def get_rank_world() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def get_output_filesystem_and_path(
    output_dir: str,
) -> Tuple[Union[pafs.S3FileSystem, pafs.LocalFileSystem], str]:
    if output_dir.startswith("s3://"):
        fs = pafs.S3FileSystem()
        path = output_dir[5:]  # strip s3://
    else:
        fs = pafs.LocalFileSystem()
        path = output_dir
    return fs, path


def get_parquet_writer(
    fs: Union[pafs.S3FileSystem, pafs.LocalFileSystem],
    path_prefix: str,
    rank: int,
    shard_idx: int,
    schema: pa.Schema,
) -> Tuple[pq.ParquetWriter, Any]:
    """Create a rank-specific parquet writer."""
    file_path = f"{path_prefix}/rank{rank}_embeddings_{shard_idx:03d}.parquet"
    sink = fs.open_output_stream(file_path)
    writer = pq.ParquetWriter(sink, schema, use_dictionary=True)
    return writer, sink


def main(cfg: DictConfig) -> None:
    local_rank = setup()
    rank, world_size = get_rank_world()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log.info(
        f"Rank {rank}/{world_size} starting on device {device} "
        f"(hostname={os.uname().nodename}, pid={os.getpid()}, time={datetime.now().isoformat()})",
    )

    log.info("Loading vocabulary and collator configuration...")
    vocab = GeneVocab.from_file(cfg.paths.vocab_file)
    coll_cfg = om.load(cfg.paths.collator_config_path)
    collator = DataCollator(
        vocab=vocab,
        do_padding=coll_cfg.get("do_padding", True),
        unexp_padding=False,
        pad_token_id=coll_cfg.pad_token_id,
        pad_value=coll_cfg.pad_value,
        do_mlm=False,
        do_binning=coll_cfg.get("do_binning", True),
        log_transform=coll_cfg.get("log_transform", False),
        target_sum=coll_cfg.get("target_sum"),
        mlm_probability=coll_cfg.mlm_probability,
        mask_value=coll_cfg.mask_value,
        max_length=cfg.data.max_length,
        sampling=coll_cfg.sampling,
        data_style="pcpt",
        num_bins=coll_cfg.get("num_bins", 51),
        right_binning=coll_cfg.get("right_binning", False),
        reserve_keys=cfg.data.reserve_keys,
    )

    log.info("Loading model checkpoint and configuration...")
    model_cfg = om.load(cfg.paths.model_config_path)
    model_cfg["attn_config"]["attn_impl"] = cfg.model.attn_impl
    model_cfg["attn_config"]["use_attn_mask"] = cfg.model.use_attn_mask

    model = ComposerSCGPTModel(model_config=model_cfg, collator_config=coll_cfg)
    torch.cuda.empty_cache()

    state = torch.load(cfg.paths.model_file, map_location=device)["state"]["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    log.info("Loading dataset and preparing DataLoader...")
    ds = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        streaming=cfg.dataset.streaming,
        cache_dir=cfg.dataset.get("cache_dir", None),
    )
    ds = ds.with_format("torch")

    if cfg.dataset.streaming:
        ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
        sampler = None
    else:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    loader = streaming.StreamingDataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=False,
    )

    total_batches: Optional[int] = None
    if sampler is not None:
        try:
            total_batches = len(sampler)
        except Exception:
            total_batches = None
    elif (not cfg.dataset.streaming) and hasattr(ds, "__len__"):
        N = len(ds)
        B = int(cfg.data.batch_size)
        total_batches = (N // B) + (1 if (N % B) else 0)

    pbar = tqdm(
        total=total_batches,
        desc=f"Rank {rank} embedding & writing",
        disable=(rank != 0),
    )

    schema = pa.schema(
        [
            pa.field("drug", pa.dictionary(pa.int32(), pa.string())),
            pa.field("sample", pa.dictionary(pa.int32(), pa.string())),
            pa.field("cell_line", pa.dictionary(pa.int32(), pa.string())),
            pa.field("BARCODE_SUB_LIB_ID", pa.string()),
            pa.field(cfg.model_name, pa.list_(pa.float32(), model_cfg["d_model"])),
        ],
    )
    fs, output_path = get_output_filesystem_and_path(cfg.paths.output_dir)

    precision = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[model_cfg["precision"]]

    use_autocast = (
        device.type == "cuda" and precision in (torch.float16, torch.bfloat16)
    ) or (device.type == "cpu" and precision is torch.bfloat16)

    # ---- Run & write ----
    row_count: int = 0
    shard_idx: int = 0
    writer: Optional[pq.ParquetWriter] = None
    sink: Optional[Any] = None

    try:
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=precision,
            enabled=use_autocast,
        ):
            for batch in loader:
                bs = batch["gene"].shape[0]

                # Rotate to a new ParquetWriter if starting a shard
                if writer is None:
                    writer, sink = get_parquet_writer(
                        fs,
                        output_path,
                        rank,
                        shard_idx,
                        schema,
                    )

                # Extract metadata
                drugs = batch["drug"]
                samples = batch["sample"]
                cells = batch["cell_line_id"]
                barcodes = batch["BARCODE_SUB_LIB_ID"]

                # Compute CLS embeddings
                ids = batch["gene"].to(device)
                expr = batch["expr"].to(device)
                mask = ~ids.eq(coll_cfg.pad_token_id)
                embs = model.model._encode(ids, expr, src_key_padding_mask=mask)
                cls_np = embs[:, 0, :].cpu().numpy()

                # Build and write Arrow Table
                table = pa.Table.from_pydict(
                    {
                        "drug": drugs,
                        "sample": samples,
                        "cell_line": cells,
                        "BARCODE_SUB_LIB_ID": barcodes,
                        cfg.model_name: [list(r) for r in cls_np],
                    },
                    schema=schema,
                )
                writer.write_table(table)

                row_count += bs
                pbar.update(bs)

                # If chunk size reached, close and advance shard
                if row_count >= cfg.parquet.chunk_size:
                    writer.close()
                    sink.close()
                    writer = None
                    sink = None
                    row_count = 0
                    shard_idx += 1
    finally:
        if writer:
            writer.close()
        if sink:
            sink.close()
        pbar.close()
        if rank == 0:
            log.info(
                f"Rank {rank} finished writing shard {shard_idx} to Parquet files.",
            )


if __name__ == "__main__":
    yaml_path: str = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg: DictConfig = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
