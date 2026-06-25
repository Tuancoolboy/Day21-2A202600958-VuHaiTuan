#!/usr/bin/env python3
"""End-to-end Lab 21 LoRA/QLoRA training script.

This script mirrors the Colab notebook flow:
1. Load or read an Alpaca-style dataset.
2. Format, clean, dedupe, tokenize, and split it.
3. Train LoRA adapters for multiple ranks.
4. Save metrics, loss curves, adapters, and qualitative comparisons.

Run on a CUDA GPU runtime, for example:
    python train_lora_qlora.py --output-dir outputs/lab21_lora_t4
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""

DEFAULT_TEST_PROMPTS = [
    "Giải thích khái niệm machine learning cho người mới bắt đầu.",
    "Viết đoạn code Python tính số Fibonacci thứ n.",
    "Liệt kê 5 nguyên tắc thiết kế UI/UX.",
    "Tóm tắt sự khác biệt giữa LoRA và QLoRA.",
    "Phân biệt prompt engineering, RAG, và fine-tuning.",
    "Khi nào nên dùng RAG thay vì fine-tuning?",
    "Giải thích cách hoạt động của Flash Attention.",
    "List 3 câu hỏi phỏng vấn cho ML Engineer junior.",
    "Cho biết ưu điểm của Transformer so với RNN.",
    "Cách evaluate performance của một LLM fine-tuned model?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare LoRA ranks for Lab 21 on a CUDA GPU."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lab21_lora_t4"))
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-bnb-4bit")
    parser.add_argument("--dataset-name", default="5CD-AI/Vietnamese-alpaca-gpt4-gg-translated")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--custom-data",
        type=Path,
        help="Optional .json or .jsonl file with instruction/input/output rows.",
    )
    parser.add_argument("--instruction-col", help="Override instruction column name.")
    parser.add_argument("--input-col", help="Override input/context column name.")
    parser.add_argument("--output-col", help="Override output/answer column name.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--eval-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-cap", type=int, default=1024)
    parser.add_argument("--min-output-tokens", type=int, default=10)

    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 8, 64])
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--eval-strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--packing", action="store_true", help="Enable SFT packing.")

    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument("--qualitative-rank", type=int, default=16)
    parser.add_argument("--num-qualitative-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--gpu-cost-usd-per-hour", type=float, default=0.35)
    return parser.parse_args()


def read_custom_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Custom .json data must be a list of objects.")
    return data


def pick_column(
    columns: Sequence[str],
    override: Optional[str],
    candidates: Sequence[str],
    required: bool,
    label: str,
) -> Optional[str]:
    if override:
        if override not in columns:
            raise ValueError(f"--{label}-col={override!r} not found in columns: {list(columns)}")
        return override
    for name in candidates:
        if name in columns:
            return name
    if required:
        raise ValueError(f"Could not detect {label} column. Available columns: {list(columns)}")
    return None


def format_example(example: Dict[str, Any], instruction_col: str, input_col: Optional[str], output_col: str) -> Dict[str, str]:
    instruction = str(example.get(instruction_col) or "").strip()
    input_text = str(example.get(input_col) or "").strip() if input_col else ""
    output = str(example.get(output_col) or "").strip()

    if input_text:
        text = ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output,
        )
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": text,
    }


def round_up_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def safe_exp(value: float) -> float:
    if value is None or math.isnan(value):
        return float("nan")
    try:
        return float(math.exp(value))
    except OverflowError:
        return float("inf")


def prepare_dataset(args: argparse.Namespace) -> Tuple[Any, Any, int, Dict[str, Any], Any]:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    import numpy as np

    if args.custom_data:
        raw = Dataset.from_list(read_custom_rows(args.custom_data))
        dataset_label = str(args.custom_data)
    else:
        raw = load_dataset(args.dataset_name, split=args.dataset_split)
        dataset_label = f"{args.dataset_name}:{args.dataset_split}"

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(raw):
        raw = raw.shuffle(seed=args.seed).select(range(args.sample_size))

    columns = raw.column_names
    instruction_col = pick_column(
        columns,
        args.instruction_col,
        ["instruction", "instruction_vi", "prompt", "question", "query"],
        required=True,
        label="instruction",
    )
    input_col = pick_column(
        columns,
        args.input_col,
        ["input", "input_vi", "context", "context_vi"],
        required=False,
        label="input",
    )
    output_col = pick_column(
        columns,
        args.output_col,
        ["output", "output_vi", "response", "answer", "completion"],
        required=True,
        label="output",
    )

    print(f"Dataset: {dataset_label}")
    print(f"Columns: instruction={instruction_col!r}, input={input_col!r}, output={output_col!r}")

    rows = []
    seen = set()
    for example in raw:
        formatted = format_example(example, instruction_col, input_col, output_col)
        if not formatted["instruction"] or not formatted["output"]:
            continue
        key = formatted["text"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(formatted)

    if len(rows) < 2:
        raise ValueError("Need at least 2 usable examples after cleaning.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    cleaned_rows = []
    for row in rows:
        output_tokens = len(tokenizer.encode(row["output"], add_special_tokens=False))
        if output_tokens >= args.min_output_tokens:
            cleaned_rows.append(row)

    if len(cleaned_rows) < 2:
        raise ValueError(
            "Need at least 2 examples after min-output-token filtering. "
            "Lower --min-output-tokens or add more data."
        )

    ds = Dataset.from_list(cleaned_rows)
    lengths = [len(tokenizer.encode(row["text"])) for row in ds]
    p50 = int(np.percentile(lengths, 50))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    max_seq_length = min(args.max_seq_cap, round_up_power_of_two(max(p95, 256)))

    split = ds.train_test_split(test_size=args.eval_size, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    stats = {
        "dataset": dataset_label,
        "raw_rows": len(raw),
        "formatted_rows": len(rows),
        "clean_rows": len(ds),
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "p50_tokens": p50,
        "p95_tokens": p95,
        "p99_tokens": p99,
        "max_seq_length": max_seq_length,
        "instruction_col": instruction_col,
        "input_col": input_col,
        "output_col": output_col,
    }

    print(
        "Token stats: "
        f"min={min(lengths)}, max={max(lengths)}, p50={p50}, p95={p95}, p99={p99}"
    )
    print(f"Using max_seq_length={max_seq_length}")
    print(f"Split: train={len(train_ds)}, eval={len(eval_ds)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 3))
        plt.hist(lengths, bins=40, color="#0E2A52", edgecolor="white")
        plt.axvline(p95, color="#C8102E", linestyle="--", label=f"p95={p95}")
        plt.axvline(max_seq_length, color="green", linestyle="--", label=f"chosen={max_seq_length}")
        plt.xlabel("Tokens")
        plt.ylabel("Count")
        plt.title("Token length distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "token_length_distribution.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"Skipping token histogram plot: {exc}")

    return train_ds, eval_ds, max_seq_length, stats, tokenizer


def patch_trainer_tokenizer_alias() -> None:
    from transformers import Trainer

    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        return
    if getattr(Trainer.__init__, "_lab21_tokenizer_alias", False):
        return

    original_init = Trainer.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        return original_init(self, *args, **kwargs)

    patched_init._lab21_tokenizer_alias = True  # type: ignore[attr-defined]
    Trainer.__init__ = patched_init  # type: ignore[assignment]

    try:
        import unsloth.models._utils as unsloth_utils

        unsloth_utils._original_trainer_init = patched_init
    except Exception:
        pass


def load_base_model(args: argparse.Namespace, max_seq_length: int, fast_language_model: Any) -> Tuple[Any, Any]:
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def wrap_with_lora(
    model: Any,
    args: argparse.Namespace,
    rank: int,
    alpha: int,
    fast_language_model: Any,
) -> Any:
    return fast_language_model.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        target_modules=args.target_modules,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )


def make_trainer(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    output_dir: Path,
    max_seq_length: int,
    args: argparse.Namespace,
) -> Any:
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    patch_trainer_tokenizer_alias()

    try:
        from trl import SFTConfig

        has_sft_config = True
    except ImportError:
        SFTConfig = None  # type: ignore[assignment]
        has_sft_config = False

    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in training_args_params else "evaluation_strategy"

    base_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "eval_accumulation_steps": 4,
        "prediction_loss_only": True,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "report_to": "none",
    }
    base_kwargs[eval_key] = args.eval_strategy

    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    supports_old_kwargs = "dataset_text_field" in sft_params

    if has_sft_config:
        sft_config_params = inspect.signature(SFTConfig.__init__).parameters  # type: ignore[union-attr]
        sft_extra = {
            "dataset_text_field": "text",
            "packing": args.packing,
            "max_seq_length": max_seq_length,
        }
        valid_base = {k: v for k, v in base_kwargs.items() if k in sft_config_params}
        valid_extra = {k: v for k, v in sft_extra.items() if k in sft_config_params}
        trainer_args = SFTConfig(**valid_base, **valid_extra)  # type: ignore[misc]
    else:
        valid_base = {k: v for k, v in base_kwargs.items() if k in training_args_params}
        trainer_args = TrainingArguments(**valid_base)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": trainer_args,
    }
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if supports_old_kwargs:
        trainer_kwargs.update(
            {
                "dataset_text_field": "text",
                "max_seq_length": max_seq_length,
                "packing": args.packing,
            }
        )
    return SFTTrainer(**trainer_kwargs)


def safe_evaluate(trainer: Any) -> float:
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    try:
        from transformers.utils.notebook import NotebookProgressCallback

        trainer.remove_callback(NotebookProgressCallback)
    except Exception:
        pass

    try:
        return float(trainer.evaluate()["eval_loss"])
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        print(f"trainer.evaluate() failed ({type(exc).__name__}); falling back to manual eval.")

    gc.collect()
    torch.cuda.empty_cache()
    model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in trainer.get_eval_dataloader():
            batch = {key: value.to(device) for key, value in batch.items() if hasattr(value, "to")}
            output = model(**batch)
            total += float(output.loss.item())
            count += 1
            del output
            torch.cuda.empty_cache()
    return total / max(count, 1)


def train_one_rank(
    rank: int,
    train_ds: Any,
    eval_ds: Any,
    max_seq_length: int,
    args: argparse.Namespace,
    fast_language_model: Any,
) -> Tuple[Dict[str, Any], Any]:
    import pandas as pd
    import torch

    alpha = rank * 2
    adapter_dir = args.output_dir / f"r{rank}"

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n========== Training r={rank}, alpha={alpha} ==========")
    base_model, tokenizer = load_base_model(args, max_seq_length, fast_language_model)
    model = wrap_with_lora(base_model, args, rank=rank, alpha=alpha, fast_language_model=fast_language_model)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.3f}%)")

    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=adapter_dir,
        max_seq_length=max_seq_length,
        args=args,
    )

    started = time.time()
    trainer.train()
    wall_seconds = time.time() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    trainer.save_model(str(adapter_dir))
    print(f"Saved adapter: {adapter_dir}")

    log_df = pd.DataFrame(trainer.state.log_history)
    if not log_df.empty:
        log_df.insert(0, "rank", rank)
        log_df.to_csv(args.output_dir / f"training_log_r{rank}.csv", index=False)

    try:
        eval_loss = safe_evaluate(trainer)
    except Exception as exc:
        print(f"Eval failed for r={rank}: {exc}")
        eval_loss = float("nan")

    metrics = {
        "model_variant": f"r{rank}",
        "rank": rank,
        "alpha": alpha,
        "trainable_params": int(trainable_params),
        "train_time_min": wall_seconds / 60,
        "peak_vram_gb": peak_vram_gb,
        "eval_loss": float(eval_loss),
        "eval_perplexity": safe_exp(float(eval_loss)),
        "adapter_dir": str(adapter_dir),
    }

    del trainer, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, log_df


def evaluate_base_model(
    train_ds: Any,
    eval_ds: Any,
    max_seq_length: int,
    args: argparse.Namespace,
    fast_language_model: Any,
) -> Dict[str, Any]:
    import torch

    print("\n========== Evaluating base model ==========")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base_model, tokenizer = load_base_model(args, max_seq_length, fast_language_model)
    trainer = make_trainer(
        model=base_model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=args.output_dir / "base_eval",
        max_seq_length=max_seq_length,
        args=args,
    )
    try:
        eval_loss = safe_evaluate(trainer)
    except Exception as exc:
        print(f"Base eval failed: {exc}")
        eval_loss = float("nan")

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    del trainer, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_variant": "base",
        "rank": None,
        "alpha": None,
        "trainable_params": 0,
        "train_time_min": 0.0,
        "peak_vram_gb": peak_vram_gb,
        "eval_loss": float(eval_loss),
        "eval_perplexity": safe_exp(float(eval_loss)),
        "adapter_dir": "",
    }


def save_loss_curve(log_frames: Iterable[Any], output_dir: Path) -> None:
    frames = [frame for frame in log_frames if frame is not None and not frame.empty]
    if not frames:
        return
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_dir / "training_logs_all_ranks.csv", index=False)
        if "loss" not in combined.columns or "step" not in combined.columns:
            return

        plt.figure(figsize=(8, 4))
        for rank, group in combined[combined["loss"].notna()].groupby("rank"):
            plt.plot(group["step"], group["loss"], label=f"r={rank}")
        plt.xlabel("Step")
        plt.ylabel("Training loss")
        plt.title("Loss Curve by LoRA Rank")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"Skipping loss curve plot: {exc}")


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    fast_language_model: Any,
) -> str:
    import torch

    fast_language_model.for_inference(model)
    text = ALPACA_TEMPLATE_NO_INPUT.format(instruction=prompt, output="")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def run_qualitative_comparison(
    args: argparse.Namespace,
    max_seq_length: int,
    fast_language_model: Any,
) -> None:
    import pandas as pd
    import torch
    from peft import PeftModel

    adapter_dir = args.output_dir / f"r{args.qualitative_rank}"
    if not adapter_dir.exists():
        print(f"Skipping qualitative comparison; adapter not found: {adapter_dir}")
        return

    prompts = DEFAULT_TEST_PROMPTS[: args.num_qualitative_prompts]
    print(f"\n========== Qualitative comparison with r={args.qualitative_rank} ==========")

    base_model, tokenizer = load_base_model(args, max_seq_length, fast_language_model)
    base_outputs = []
    for prompt in prompts:
        base_outputs.append(
            generate_response(base_model, tokenizer, prompt, args.max_new_tokens, fast_language_model)
        )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    ft_base, ft_tokenizer = load_base_model(args, max_seq_length, fast_language_model)
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_dir))
    ft_outputs = []
    for prompt in prompts:
        ft_outputs.append(
            generate_response(ft_model, ft_tokenizer, prompt, args.max_new_tokens, fast_language_model)
        )

    rows = []
    for prompt, base, finetuned in zip(prompts, base_outputs, ft_outputs):
        rows.append(
            {
                "prompt": prompt,
                "base": base,
                f"finetuned_r{args.qualitative_rank}": finetuned,
            }
        )

    qual_df = pd.DataFrame(rows)
    qual_df.to_csv(args.output_dir / "qualitative_comparison.csv", index=False)
    print(f"Saved qualitative comparison: {args.output_dir / 'qualitative_comparison.csv'}")

    del ft_model, ft_base, ft_tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def write_run_summary(args: argparse.Namespace, dataset_stats: Dict[str, Any], metrics: List[Dict[str, Any]]) -> None:
    train_minutes = sum(row["train_time_min"] for row in metrics if row["model_variant"] != "base")
    estimated_cost = train_minutes / 60 * args.gpu_cost_usd_per_hour
    summary = {
        "model_name": args.model_name,
        "target_modules": args.target_modules,
        "ranks": args.ranks,
        "dataset": dataset_stats,
        "total_train_time_min": train_minutes,
        "gpu_cost_usd_per_hour": args.gpu_cost_usd_per_hour,
        "estimated_cost_usd": estimated_cost,
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Total train time: {train_minutes:.1f} min")
    print(f"Estimated cost: ${estimated_cost:.2f} at ${args.gpu_cost_usd_per_hour}/hr")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Import Unsloth before Transformers-heavy training code so kernels are patched early.
    from unsloth import FastLanguageModel
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. In Colab, set Runtime > Change runtime type > GPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Output dir: {args.output_dir}")

    train_ds, eval_ds, max_seq_length, dataset_stats, _ = prepare_dataset(args)

    metrics: List[Dict[str, Any]] = []
    if not args.skip_base_eval:
        metrics.append(evaluate_base_model(train_ds, eval_ds, max_seq_length, args, FastLanguageModel))

    log_frames = []
    for rank in args.ranks:
        rank_metrics, log_df = train_one_rank(
            rank=rank,
            train_ds=train_ds,
            eval_ds=eval_ds,
            max_seq_length=max_seq_length,
            args=args,
            fast_language_model=FastLanguageModel,
        )
        metrics.append(rank_metrics)
        log_frames.append(log_df)

    summary_df = pd.DataFrame(metrics)
    rank_sort = summary_df["rank"].fillna(-1).astype(int)
    summary_df = summary_df.assign(_rank_sort=rank_sort).sort_values("_rank_sort").drop(columns=["_rank_sort"])
    summary_df.to_csv(args.output_dir / "rank_experiment_summary.csv", index=False)
    print("\n=== Rank Experiment Summary ===")
    print(summary_df.to_string(index=False))

    save_loss_curve(log_frames, args.output_dir)
    if not args.skip_qualitative:
        run_qualitative_comparison(args, max_seq_length, FastLanguageModel)
    write_run_summary(args, dataset_stats, metrics)

    print("\nDone. Key outputs:")
    print(f"- {args.output_dir / 'rank_experiment_summary.csv'}")
    print(f"- {args.output_dir / 'qualitative_comparison.csv'}")
    print(f"- {args.output_dir / 'loss_curve.png'}")
    adapter_dirs = ", ".join(str(args.output_dir / f"r{rank}") for rank in args.ranks)
    print(f"- {adapter_dirs}")


if __name__ == "__main__":
    main()
