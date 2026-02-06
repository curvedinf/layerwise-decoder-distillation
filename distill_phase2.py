#!/usr/bin/env python3
"""
distill_phase2.py

Phase 2 knowledge distillation: train the full student model to match
teacher logits using KL divergence.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

from distill import (
    BETAS,
    GRAD_ACCUM_STEPS,
    HF_TRUST_REMOTE_CODE,
    LOG_EVERY,
    SAVE_EVERY,
    SEED,
    TRAIN_GLOB,
    TRY_FUSED_ADAMW,
    WEIGHT_DECAY,
    _ddp_cleanup,
    _ddp_setup,
    _find_latest_student_ckpt,
    _hf_load_path,
    _is_master,
    _resolve_run_dir,
    _save_checkpoint,
    _student_ckpt_name,
    load_state_dict_strict,
    swap_hf_decoders,
    CachedCheckpoint,
    DistributedDataLoader,
    STUDENT_DECODER_KIND,
    STUDENT_DECODER_KWARGS,
)

# Phase 2 defaults (aligned with RADLADS-style KD)
PHASE2_LR = 1e-5
DEVICE_BATCH_SIZE = 16
SEQ_LEN = 512


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    s_logprob = F.log_softmax(student_logits.float(), dim=-1)
    t_logprob = F.log_softmax(teacher_logits.float(), dim=-1)
    return F.kl_div(s_logprob, t_logprob, log_target=True, reduction="batchmean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="runs", help="Output directory for logs/checkpoints.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of phase-2 steps to run.")
    parser.add_argument("--batch-size", type=int, default=DEVICE_BATCH_SIZE, help="Per-device batch size.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save student checkpoint every N steps.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory to resume/save into.")
    parser.add_argument("--new-run", action="store_true", help="Force a new run directory under --out-dir.")
    parser.add_argument("--memory-cache", action="store_true", help="Keep checkpoints in memory instead of writing to disk.")
    parser.add_argument("--rank", type=int, default=0, help="Process rank (for multi-process runs).")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0, help="Local rank.")
    parser.add_argument("--world-size", type=int, default=1, help="World size (number of processes).")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Master address for DDP.")
    parser.add_argument("--master-port", type=int, default=29500, help="Master port for DDP.")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    rank, local_rank, world_size, device = _ddp_setup(
        args.rank,
        args.local_rank,
        args.world_size,
        args.master_addr,
        args.master_port,
    )
    master = _is_master(rank)

    out_dir, run_id = _resolve_run_dir(args.out_dir, args.run_dir, args.new_run)
    if master:
        os.makedirs(out_dir, exist_ok=True)
    memory_cache: list[CachedCheckpoint] = []

    hf_path = _hf_load_path()
    if master:
        print(f"[hf] base checkpoint: {hf_path}")

    teacher = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    ).to(device)
    swap_hf_decoders(student, decoder_kind=STUDENT_DECODER_KIND, decoder_kwargs=STUDENT_DECODER_KWARGS)

    if dist.is_initialized():
        # Full-model training: standard DDP path.
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[local_rank])

    train_loader = DistributedDataLoader(TRAIN_GLOB, args.batch_size, args.seq_len, rank, world_size)

    def _make_optimizer() -> torch.optim.Optimizer:
        kwargs = dict(lr=PHASE2_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        if TRY_FUSED_ADAMW:
            try:
                return torch.optim.AdamW(student.parameters(), **kwargs, fused=True)
            except TypeError:
                pass
        return torch.optim.AdamW(student.parameters(), **kwargs)

    opt = _make_optimizer()

    resume_path, resume_step = _find_latest_student_ckpt(out_dir)
    global_step = 0
    phase2_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        student_target = student.module if hasattr(student, "module") else student
        load_state_dict_strict(student_target, ckpt.get("student", {}))
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", resume_step or 0))
        phase2_step = int(ckpt.get("phase2_step", 0))
        if master:
            print(f"[resume] {resume_path} (global_step={global_step}, phase2_step={phase2_step})")

    if master:
        meta = dict(
            run_id=run_id,
            hf_model_path=hf_path,
            phase2_steps=int(args.steps),
            device_batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
            grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=PHASE2_LR,
            weight_decay=WEIGHT_DECAY,
            world_size=world_size,
            save_every=int(args.save_every),
            resume_from=resume_path,
        )
        with open(os.path.join(out_dir, "meta_phase2.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        tokens_per_step = args.batch_size * args.seq_len * world_size * GRAD_ACCUM_STEPS
        print(f"[run] out_dir={out_dir}")
        print(f"[run] world_size={world_size} phase2_steps={args.steps} tokens/step={tokens_per_step}")

    t0 = time.time()
    try:
        for step in range(phase2_step, int(args.steps)):
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0

            for _ in range(GRAD_ACCUM_STEPS):
                x, _ = train_loader.next_batch(device)
                with torch.no_grad():
                    t_logits = teacher(x).logits
                s_logits = student(x).logits
                loss = distill_loss(s_logits, t_logits)
                (loss / GRAD_ACCUM_STEPS).backward()
                loss_acc += float(loss.detach().item())

            opt.step()
            global_step += 1
            phase2_step += 1

            if args.save_every and args.save_every > 0 and (global_step % int(args.save_every) == 0):
                if master:
                    ckpt = dict(
                        run_id=run_id,
                        hf_model_path=hf_path,
                        global_step=global_step,
                        phase2_step=phase2_step,
                        student=student.module.state_dict() if hasattr(student, "module") else student.state_dict(),
                        optimizer=opt.state_dict(),
                    )
                    _save_checkpoint(
                        ckpt,
                        _student_ckpt_name(global_step),
                        out_dir,
                        memory_cache,
                        args.memory_cache,
                        kind="step",
                    )

            if master and (step % LOG_EVERY == 0 or step + 1 == int(args.steps)):
                elapsed = time.time() - t0
                tokens_per_step = args.batch_size * args.seq_len * world_size * GRAD_ACCUM_STEPS
                tok_s = (global_step * tokens_per_step) / max(1e-9, elapsed)
                avg_loss = loss_acc / GRAD_ACCUM_STEPS
                print(
                    f"[train] step={phase2_step}/{args.steps} global={global_step} "
                    f"loss={avg_loss:.4f} tok/s={tok_s:.0f}"
                )

        if master:
            ckpt = dict(
                run_id=run_id,
                hf_model_path=hf_path,
                global_step=global_step,
                phase2_step=phase2_step,
                student=student.module.state_dict() if hasattr(student, "module") else student.state_dict(),
                optimizer=opt.state_dict(),
            )
            _save_checkpoint(
                ckpt,
                _student_ckpt_name(global_step),
                out_dir,
                memory_cache,
                args.memory_cache,
                force_disk=True,
                kind="final",
            )
    finally:
        _ddp_cleanup()


if __name__ == "__main__":
    main()
