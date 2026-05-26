from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.conda_env import ensure_conda_env

ensure_conda_env()

from src.train.dialogue_text_audio_teacher_student_runner import (
    S6_TEXT_AUDIO_TEACHER_STUDENT_METHODS,
    run_s6_text_audio_teacher_student_experiment,
)


DEFAULT_METHODS = ["s6_text_student_ta_teacher", "s6_text_student_ta_teacher_sa"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S6 text-only student distilled from text+audio teacher.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dialogue_task_stl_v2.yaml"))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--teacher-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--lambda-ta-kd", type=float, default=None)
    parser.add_argument("--lambda-ta-rel", type=float, default=None)
    parser.add_argument("--lambda-kd", type=float, default=None)
    parser.add_argument("--lambda-rel", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--backbone", dest="text_model_path", default=None, help="Text backbone name or local path, e.g. bert-base-uncased.")
    parser.add_argument("--text-model-path", dest="text_model_path", default=None, help="Alias for --backbone.")
    parser.add_argument("--audio-encoder", dest="audio_encoder_type", choices=["raw", "pretrained"], default=None)
    parser.add_argument("--audio-backbone", dest="audio_model_path", default=None, help="Audio backbone path, e.g. local wav2vec2/HuBERT.")
    parser.add_argument("--audio-model-path", dest="audio_model_path", default=None, help="Alias for --audio-backbone.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--audio-sample-rate", type=int, default=None)
    parser.add_argument("--audio-cache-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-id", default=None, help="Physical GPU id to expose, e.g. 3. Overrides --device to cuda.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda"

    unknown = sorted(set(args.methods) - S6_TEXT_AUDIO_TEACHER_STUDENT_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Expected subset of {sorted(S6_TEXT_AUDIO_TEACHER_STUDENT_METHODS)}")

    train_overrides = {
        "seed": args.seed,
        "epochs": args.epochs,
        "teacher_epochs": args.teacher_epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "teacher_lr": args.teacher_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "lambda_ta_kd": args.lambda_ta_kd,
        "lambda_ta_rel": args.lambda_ta_rel,
        "lambda_kd": args.lambda_kd,
        "lambda_rel": args.lambda_rel,
        "temperature": args.temperature,
        "text_model_path": args.text_model_path,
        "audio_encoder_type": args.audio_encoder_type,
        "audio_model_path": args.audio_model_path,
        "max_length": args.max_length,
        "max_audio_seconds": args.max_audio_seconds,
        "audio_sample_rate": args.audio_sample_rate,
        "audio_cache_root": args.audio_cache_root,
        "device": args.device,
    }
    if args.fp16:
        train_overrides["fp16"] = True
    if args.no_fp16:
        train_overrides["fp16"] = False

    for method in args.methods:
        method_run_name = f"{args.run_name}_{method}" if args.run_name and len(args.methods) > 1 else args.run_name
        result_path = run_s6_text_audio_teacher_student_experiment(
            args.config,
            method,
            run_name=method_run_name,
            train_overrides=train_overrides,
        )
        print(f"{method}: {result_path}")


if __name__ == "__main__":
    main()
