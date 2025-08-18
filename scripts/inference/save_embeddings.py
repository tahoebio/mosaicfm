# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import contextlib
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union

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


def list_rank_shards(
    fs: pafs.FileSystem,
    base_path: str,
    rank: int,
) -> Tuple[dict, dict]:
    """List final and in-progress shard files for a rank."""
    sel = pafs.FileSelector(base_path, recursive=False)
    infos = fs.get_file_info(sel)
    pat_final = re.compile(rf"^rank{rank}_embeddings_(\d+)\.parquet$")
    pat_tmp = re.compile(rf"^rank{rank}_embeddings_(\d+)\.parquet\.inprogress$")
    finals, tmps = {}, {}
    for info in infos:
        if info.type != pafs.FileType.File:
            continue
        name = os.path.basename(info.path)
        m = pat_final.match(name)
        if m:
            finals[int(m.group(1))] = info.path
            continue
        m = pat_tmp.match(name)
        if m:
            tmps[int(m.group(1))] = info.path
    return finals, tmps


def parquet_num_rows(fs: pafs.FileSystem, path: str) -> int:
    """Return number of rows in a parquet file (0 if unreadable)."""
    try:
        # Option A: let pyarrow open via filesystem (works for S3 and local)
        pf = pq.ParquetFile(path, filesystem=fs)
        md = pf.metadata
        return md.num_rows if md is not None else 0
    except Exception as e1:
        # Fallback: explicitly open a RandomAccessFile if available
        try:
            with fs.open_input_file(path) as f:  # seekable on S3/Local
                pf = pq.ParquetFile(f)
                md = pf.metadata
                return md.num_rows if md is not None else 0
        except Exception as e2:
            log.warning(f"Could not read rows from {path}: {e1 or e2}")
            return 0


def delete_if_exists(fs: pafs.FileSystem, path: Optional[str]) -> None:
    if not path:
        return
    try:
        fs.delete_file(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Failed to delete {path}: {e}")


def compute_resume_state(
    fs: pafs.FileSystem,
    out_path: str,
    rank: int,
    chunk_size: int,
    policy: str = "truncate_incomplete",
) -> Tuple[int, int, int, str]:
    """Inspect existing files and return resume information."""
    finals, tmps = list_rank_shards(fs, out_path, rank)

    # Clean up any leftover in-progress files
    for tmp_path in tmps.items():
        log.info(f"[rank {rank}] Removing stale in-progress file: {tmp_path}")
        delete_if_exists(fs, tmp_path)

    if not finals:
        return 0, 0, 0, "new"

    max_idx = max(finals.keys())
    resume_rows_total = 0
    for i in range(max_idx):
        if i in finals:
            resume_rows_total += chunk_size
    last_rows = parquet_num_rows(fs, finals[max_idx])
    is_complete = last_rows >= chunk_size

    if is_complete:
        resume_rows_total += chunk_size
        return resume_rows_total, max_idx + 1, 0, "new"

    if policy == "truncate_incomplete":
        log.info(
            f"[rank {rank}] Deleting partial shard {finals[max_idx]} (rows={last_rows}).",
        )
        delete_if_exists(fs, finals[max_idx])
        return resume_rows_total, max_idx, 0, "recreate"
    elif policy == "keep":
        resume_rows_total += last_rows
        return resume_rows_total, max_idx + 1, 0, "new"
    else:
        raise ValueError(f"Unknown resume.policy={policy}")


def next_writer(
    fs: pafs.FileSystem,
    base_path: str,
    rank: int,
    shard_idx: int,
    schema: pa.Schema,
) -> Tuple[pq.ParquetWriter, Any, str, str]:
    """Open a writer to a temp file; caller must later move to final name."""
    final_name = f"rank{rank}_embeddings_{shard_idx:03d}.parquet"
    final_path = f"{base_path}/{final_name}"
    tmp_path = final_path + ".inprogress"
    sink = fs.open_output_stream(tmp_path)
    writer = pq.ParquetWriter(sink, schema, use_dictionary=True)
    return writer, sink, tmp_path, final_path


def main(cfg: DictConfig) -> None:
    local_rank = setup()
    rank, world_size = get_rank_world()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
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

    # Resume inspection
    resume_enabled = bool(
        getattr(cfg, "resume", None) and cfg.resume.get("enabled", True),
    )
    resume_policy = (
        cfg.resume.get("policy", "truncate_incomplete")
        if resume_enabled
        else "truncate_incomplete"
    )

    processed_rows_already, shard_idx, row_count, action = (0, 0, 0, "new")
    if resume_enabled:
        processed_rows_already, shard_idx, row_count, action = compute_resume_state(
            fs,
            output_path,
            rank,
            cfg.parquet.chunk_size,
            policy=resume_policy,
        )
        log.info(
            f"[rank {rank}] resume: rows_done={processed_rows_already}, "
            f"start_shard={shard_idx}, start_row_count={row_count}, action={action}",
        )

    # Handle skipping rows for "keep" policy
    skip_rows_remaining = 0
    if resume_enabled and resume_policy == "keep":
        finals, _ = list_rank_shards(fs, output_path, rank)
        last_idx = max(finals.keys()) if finals else None
        if last_idx is not None:
            skip_rows_remaining = parquet_num_rows(fs, finals[last_idx])
            log.info(
                f"[rank {rank}] Keeping partial shard with {skip_rows_remaining} rows; will skip these from dataloader.",
            )

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

    if not resume_enabled:
        shard_idx: int = 0
    current_rows_in_shard: int = 0
    writer: Optional[pq.ParquetWriter] = None
    sink: Optional[Any] = None
    tmp_path: Optional[str] = None
    final_path: Optional[str] = None

    try:
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=precision,
            enabled=use_autocast,
        ):
            for batch in loader:
                bs = batch["gene"].shape[0]

                # If keeping a partial shard, skip rows already persisted
                if skip_rows_remaining > 0:
                    to_skip = min(skip_rows_remaining, bs)
                    skip_rows_remaining -= to_skip
                    pbar.update(to_skip)
                    if to_skip == bs:
                        continue
                    # If partial-batch skip would be needed, we simply continue to next iter.

                # Rotate to a new ParquetWriter if starting a shard
                if writer is None:
                    writer, sink, tmp_path, final_path = next_writer(
                        fs,
                        output_path,
                        rank,
                        shard_idx,
                        schema,
                    )
                    current_rows_in_shard = 0

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

                current_rows_in_shard += bs
                pbar.update(bs)

                # Rotate shard if needed
                if current_rows_in_shard >= cfg.parquet.chunk_size:
                    writer.close()
                    sink.close()
                    # Atomically finalize the file
                    if tmp_path and final_path:
                        fs.move(tmp_path, final_path)
                    writer = sink = tmp_path = final_path = None
                    shard_idx += 1
                    current_rows_in_shard = 0
        # Close any last partial shard cleanly
        if writer is not None:
            writer.close()
            sink.close()
            # Promote in-progress to final even if partial
            if tmp_path and final_path:
                fs.move(tmp_path, final_path)
            writer = sink = tmp_path = final_path = None
    finally:
        # Cleanup if crashed mid-write
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        if sink is not None:
            with contextlib.suppress(Exception):
                sink.close()
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                delete_if_exists(fs, tmp_path)
        pbar.close()
        if rank == 0:
            log.info(
                f"Rank {rank} finished writing shard {shard_idx} to Parquet files.",
            )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    yaml_path: str = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg: DictConfig = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
