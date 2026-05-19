"""CLI entry point for SimPO training.

Mirrors openrlhf.cli.train_dpo but:
  * does not load a reference model (SimPO is reference-free)
  * adds --model.gamma for the target reward margin
  * uses openrlhf.trainer.simpo_trainer.SimPOTrainer
"""

import argparse
import math
import os
from datetime import datetime

from openrlhf.datasets import RewardDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor
from openrlhf.trainer.simpo_trainer import SimPOTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    model = Actor(
        args.model.model_name_or_path,
        attn_implementation=args.ds.attn_implementation,
        experts_implementation=args.ds.experts_implementation,
        param_dtype=args.ds.param_dtype,
        load_in_4bit=args.ds.load_in_4bit,
        lora_rank=args.ds.lora.rank,
        lora_alpha=args.ds.lora.alpha,
        lora_dropout=args.ds.lora.dropout,
        target_modules=args.ds.lora.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.ds.packing_samples,
        use_liger_kernel=args.ds.use_liger_kernel,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(
        args.model.model_name_or_path, model.model, "right", strategy, use_fast=not args.data.disable_fast_tokenizer
    )
    strategy.print(model)

    # NOTE: SimPO is reference-free; we do NOT load a reference model.

    # gradient_checkpointing
    if args.model.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.model.gradient_checkpointing_reentrant}
        )

    # prepare for data and dataset
    train_data = blending_datasets(
        args.data.dataset,
        args.data.dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.dataset_split,
    )

    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.data.max_len,
        strategy,
        input_template=args.data.input_template,
        is_dpo=True,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.train.micro_batch_size,
        True,
        True,
        train_dataset.collate_fn,
        num_workers=args.data.dataloader_num_workers,
    )

    eval_dataset = None
    eval_dataloader = None
    if getattr(args.eval, "dataset", None):
        eval_data = blending_datasets(
            args.eval.dataset,
            None,
            strategy,
            dataset_split=args.eval.split,
        )
        eval_dataset = RewardDataset(
            eval_data,
            tokenizer,
            args.data.max_len,
            strategy,
            input_template=args.data.input_template,
            is_dpo=True,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.train.micro_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train.batch_size
    max_steps = math.ceil(args.train.max_epochs * num_update_steps_per_epoch)

    cfg = dict(
        optim=args.optim,
        muon=vars(args.muon),
        adam=vars(args.adam),
        lr_scheduler=args.lr_scheduler,
        lr_warmup_ratio=args.lr_warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_norm=args.max_norm,
        scheduler_steps=max_steps,
    )
    # strategy.prepare only gets the policy; no ref_model second arg.
    (model, optim, scheduler) = strategy.prepare((model, cfg))

    # load checkpoint
    consumed_samples = 0
    if args.ckpt.load_enable and os.path.exists(args.ckpt.path):
        load_path, states = strategy.load_ckpt(model.model, args.ckpt.path)
        if load_path is not None:
            consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {args.ckpt.path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.ckpt.output_dir, exist_ok=True)

    trainer = SimPOTrainer(
        model=model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.model.beta,
        gamma=args.model.gamma,
        max_epochs=args.train.max_epochs,
        save_hf_ckpt=args.ckpt.save_hf,
        disable_ds_ckpt=args.ckpt.disable_ds,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.ckpt.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--ckpt.output_dir", type=str, default="./ckpt")
    parser.add_argument("--ckpt.save_steps", type=int, default=-1)
    parser.add_argument("--ckpt.save_hf", action="store_true", default=False)
    parser.add_argument("--ckpt.disable_ds", action="store_true", default=False)
    parser.add_argument("--logger.logging_steps", type=int, default=1)
    parser.add_argument("--eval.steps", type=int, default=-1)
    parser.add_argument("--ckpt.path", type=str, default="./ckpt/checkpoints_simpo")
    parser.add_argument("--ckpt.max_num", type=int, default=3)
    parser.add_argument("--ckpt.max_mem", type=int, default=int(1e8))
    parser.add_argument("--ds.use_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--train.micro_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train.batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--ckpt.load_enable", action="store_true", default=False)
    parser.add_argument("--model.gradient_checkpointing_enable", action="store_true", default=False)
    parser.add_argument("--ds.deepcompile", action="store_true", default=False)
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument("--train.full_determinism_enable", action="store_true", default=False)
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--data.dataloader_num_workers", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--ds.zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--ds.param_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--ds.zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--ds.adam_offload", action="store_true", default=False)
    parser.add_argument(
        "--ds.attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, sdpa)",
    )
    parser.add_argument(
        "--ds.experts_implementation",
        type=str,
        default=None,
        choices=["eager", "batched_mm", "grouped_mm", "deepgemm"],
    )
    parser.add_argument("--ds.use_liger_kernel", action="store_true", default=False)
    parser.add_argument("--ds.grad_accum_dtype", type=str, default=None)
    parser.add_argument("--ds.overlap_comm", action="store_true", default=False)
    parser.add_argument("--model.gradient_checkpointing_reentrant", action="store_true", default=False)
    parser.add_argument("--ds.tensor_parallel_size", type=int, default=1)

    # SimPO
    parser.add_argument("--train.max_epochs", type=int, default=1)
    parser.add_argument(
        "--model.beta",
        type=float,
        default=2.0,
        help="SimPO beta. Larger than DPO due to length normalization. Paper-recommended 2.0~2.5.",
    )
    parser.add_argument(
        "--model.gamma",
        type=float,
        default=0.0,
        help="SimPO target reward margin. 0 disables margin term. Paper-recommended 0.5~1.5.",
    )
    # Kept for parity with DPO trainer args; SimPO trainer reads these via strategy.args
    parser.add_argument("--model.aux_loss_coef", type=float, default=0, help="MoE balancing loss coef")
    parser.add_argument("--model.nll_loss_coef", type=float, default=0, help="Optional NLL regularization coef")
    # DPO-only flags retained (no-op for SimPO) so shared launcher scripts don't error
    parser.add_argument("--model.ipo_enable", action="store_true", default=False)
    parser.add_argument("--model.label_smoothing", type=float, default=0.0)

    # Optimizer + scheduler + grad clip
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "muon"])
    parser.add_argument("--muon.lr", type=float, default=0.02)
    parser.add_argument("--muon.momentum", type=float, default=0.95)
    parser.add_argument("--muon.ns_steps", type=int, default=5)
    parser.add_argument("--muon.nesterov", action="store_true", default=True)
    parser.add_argument("--muon.no_nesterov", dest="muon.nesterov", action="store_false")
    parser.add_argument("--adam.lr", type=float, default=1e-6, help="SimPO typically uses smaller LR than DPO.")
    parser.add_argument("--adam.betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--adam.eps", type=float, default=1e-8)
    parser.add_argument("--adam.weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")

    # Context Parallel
    parser.add_argument("--ds.ring_attn_size", type=int, default=1)
    parser.add_argument("--ds.ring_attn_head_stride", type=int, default=1)

    # LoRA
    parser.add_argument("--ds.load_in_4bit", action="store_true", default=False)
    parser.add_argument("--ds.lora.rank", type=int, default=0)
    parser.add_argument("--ds.lora.alpha", type=int, default=16)
    parser.add_argument("--ds.lora.target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--ds.lora.dropout", type=float, default=0)

    # packing
    parser.add_argument("--ds.packing_samples", action="store_true", default=False)

    # dataset
    parser.add_argument("--model.model_name_or_path", type=str, default=None)
    parser.add_argument("--data.dataset", type=str, default=None)
    parser.add_argument("--data.dataset_probs", type=str, default=None)
    parser.add_argument("--eval.dataset", type=str, default=None)
    parser.add_argument("--data.dataset_split", type=str, default="train")
    parser.add_argument("--eval.split", type=str, default="test")
    parser.add_argument("--data.max_samples", type=int, default=1000000)
    parser.add_argument("--data.prompt_key", type=str, default=None)
    parser.add_argument("--data.chosen_key", type=str, default="chosen")
    parser.add_argument("--data.rejected_key", type=str, default="rejected")
    parser.add_argument("--data.input_template", type=str, default=None)
    parser.add_argument("--data.apply_chat_template", action="store_true", default=False)
    parser.add_argument("--data.tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--data.max_len", type=int, default=512)

    # wandb / tensorboard
    parser.add_argument("--logger.wandb.key", type=str, default=None)
    parser.add_argument("--logger.wandb.org", type=str, default=None)
    parser.add_argument("--logger.wandb.group", type=str, default=None)
    parser.add_argument("--logger.wandb.project", type=str, default="openrlhf_train_simpo")
    parser.add_argument(
        "--logger.wandb.run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    parser.add_argument("--logger.tensorboard_dir", type=str, default=None)

    # ModelScope
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()
    from openrlhf.utils.config import hierarchize

    args = hierarchize(args)

    if args.data.input_template and "{}" not in args.data.input_template:
        print("[Warning] '{}' not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.data.input_template and "\\n" in args.data.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ds.ring_attn_size > 1:
        assert args.ds.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.ds.packing_samples and "flash_attention" not in args.ds.attn_implementation:
        print(
            "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
        )
        args.ds.attn_implementation = "flash_attention_2"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        patch_hub()

    train(args)
