# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import logging
import os
import sys

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import streaming
import torch
import torch.distributed as dist
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


def setup():
    """Initialize distributed process group and set device."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def get_output_filesystem_and_path(output_dir: str):
    """Get filesystem and path for output directory."""
    if output_dir.startswith("s3://"):
        fs = pafs.S3FileSystem()
        path = output_dir[5:]  # strip s3://
    else:
        fs = pafs.LocalFileSystem()
        path = output_dir
    return fs, path


def get_parquet_writer(fs, path_prefix, rank, shard_idx, schema):
    """Create a rank-specific parquet writer."""
    file_path = f"{path_prefix}/rank{rank}_embeddings_{shard_idx:03d}.parquet"
    sink = fs.open_output_stream(file_path)
    writer = pq.ParquetWriter(sink, schema, use_dictionary=True)
    return writer, sink


def main(cfg: DictConfig) -> None:
    """
    Main entrypoint: load model, dataset, compute embeddings, and write chunked Parquet shards.
    """
    # Initialize distributed processing
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log.info(f"Rank {rank} started with local_rank {local_rank}...")

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

    # Set per-rank device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Coordinated model loading
    dist.barrier()
    state = torch.load(cfg.paths.model_file, map_location=device)["state"]["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    dist.barrier()

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    log.info("Loading dataset and preparing DataLoader...")
    ds = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        streaming=cfg.dataset.streaming,
        cache_dir=cfg.dataset.get("cache_dir", None),
    )
    ds = ds.with_format("torch")

    # Create distributed sampler
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)

    loader = streaming.StreamingDataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=True,
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

    # Setup output filesystem and paths
    fs, output_path = get_output_filesystem_and_path(cfg.paths.output_dir)

    row_count = 0
    shard_idx = 0
    writer = None
    sink = None
    pbar = tqdm(
        total=len(sampler),
        desc=f"Rank {rank} embedding & writing",
        disable=(rank != 0),
    )

    precision = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[model_cfg["precision"]]

    with torch.no_grad(), torch.amp.autocast(
        enabled=True,
        dtype=precision,
        device_type=device.type,
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
            embs = model.module.model._encode(ids, expr, src_key_padding_mask=mask)
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

    # Final close
    if writer:
        writer.close()
    if sink:
        sink.close()
    pbar.close()

    # Ensure all ranks finish before cleanup
    dist.barrier()
    cleanup()

    if rank == 0:
        log.info(f"Finished writing embeddings to: {cfg.paths.output_dir}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
