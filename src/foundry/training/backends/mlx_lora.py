"""MLX LoRA training backend — local fine-tuning on Apple Silicon."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from foundry.training.backends.protocol import TrainingJob
from foundry.factory.data_factory import TrainingExample


@dataclass
class MLXLoRAConfig:
    """Configuration for MLX LoRA fine-tuning."""
    # Model
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"    # HuggingFace model ID
    # LoRA hyperparams
    num_layers: int = 8              # layers to fine-tune (lower = faster, less memory)
    batch_size: int = 1              # 1 for 16GB M1 Pro
    iters: int = 100                 # training iterations
    learning_rate: float = 1e-4
    # Sequence
    max_seq_length: int = 512        # keep short for 16GB RAM
    # Output
    output_dir: str = os.path.expanduser("~/agent-foundry/.foundry/training")
    adapter_path: str = "adapters"   # relative to output_dir
    # Validation
    val_split: float = 0.1           # fraction for validation
    steps_per_eval: int = 50
    steps_per_report: int = 10
    # Optimization
    grad_checkpoint: bool = False    # enable if OOM
    mask_prompt: bool = True         # only train on response tokens


class MLXLoRABackend:
    """
    Local LoRA fine-tuning using Apple's MLX framework.

    Implements FineTuneBackend protocol.

    Workflow:
      1. launch() — prepares data, starts training, returns TrainingJob
      2. status() — checks if training is complete
      3. get_model_id() — returns adapter path for inference

    Usage::

        backend = MLXLoRABackend(config=MLXLoRAConfig(iters=50))
        job = backend.launch(dataset_path="train.jsonl", config={})
        assert job.status == "complete"
        adapter = backend.get_model_id(job.job_id)
    """

    def __init__(self, config: Optional[MLXLoRAConfig] = None) -> None:
        self.config = config or MLXLoRAConfig()
        self._python = self._find_python()
        self._jobs: dict[str, TrainingJob] = {}
        self._job_metadata: dict[str, dict[str, Any]] = {}

    # ── FineTuneBackend protocol implementation ───────────────────────

    def launch(self, dataset_path: str, config: dict[str, Any] = None) -> TrainingJob:
        """
        Launch a LoRA fine-tune job.

        Args:
            dataset_path: Path to JSONL training data (chat format).
            config: Optional overrides (iters, batch_size, num_layers, etc.)

        Returns:
            TrainingJob with status "complete" or "failed" (MLX runs synchronously).
        """
        job_id = str(uuid.uuid4())[:8]
        cfg = self.config
        if config:
            for k, v in config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        output_dir = Path(cfg.output_dir) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data directory
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        self._prepare_data_from_jsonl(dataset_path, data_dir)

        # Build command
        adapter_out = output_dir / cfg.adapter_path
        adapter_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._python, "-m", "mlx_lm", "lora",
            "--model", cfg.base_model,
            "--data", str(data_dir),
            "--train",
            "--batch-size", str(cfg.batch_size),
            "--iters", str(cfg.iters),
            "--num-layers", str(cfg.num_layers),
            "--max-seq-length", str(cfg.max_seq_length),
            "--adapter-path", str(adapter_out),
            "--steps-per-report", str(cfg.steps_per_report),
            "--steps-per-eval", str(cfg.steps_per_eval),
        ]
        if cfg.mask_prompt:
            cmd.append("--mask-prompt")
        if cfg.grad_checkpoint:
            cmd.append("--grad-checkpoint")

        print(f"[MLXLoRA] Job {job_id}: {cfg.iters} iters, "
              f"model={cfg.base_model}, layers={cfg.num_layers}")

        # Run training (synchronous — MLX trains fast on Apple Silicon)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(output_dir),
            )

            if proc.returncode != 0:
                job = TrainingJob(
                    job_id=job_id,
                    backend="mlx_lora",
                    status="failed",
                    base_model=cfg.base_model,
                    metadata={"error": proc.stderr[-2000:]},
                )
            else:
                train_loss, val_loss, iters_done = self._parse_output(proc.stdout)
                job = TrainingJob(
                    job_id=job_id,
                    backend="mlx_lora",
                    status="complete",
                    model_id=str(adapter_out),
                    base_model=cfg.base_model,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    iters_completed=iters_done,
                    metadata={
                        "adapter_path": str(adapter_out),
                        "output_dir": str(output_dir),
                    },
                )
                print(f"[MLXLoRA] Job {job_id} complete: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        except subprocess.TimeoutExpired:
            job = TrainingJob(
                job_id=job_id,
                backend="mlx_lora",
                status="failed",
                metadata={"error": "Training timed out (1h)"},
            )
        except Exception as e:
            job = TrainingJob(
                job_id=job_id,
                backend="mlx_lora",
                status="failed",
                metadata={"error": str(e)},
            )

        self._jobs[job_id] = job
        return job

    def status(self, job_id: str) -> TrainingJob:
        """Return job status. MLX training is synchronous, so always terminal."""
        if job_id not in self._jobs:
            return TrainingJob(
                job_id=job_id,
                backend="mlx_lora",
                status="failed",
                metadata={"error": f"Unknown job: {job_id}"},
            )
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> None:
        """Cancel is a no-op for MLX (synchronous training)."""
        pass

    def get_model_id(self, job_id: str) -> str:
        """Return the adapter path once training is complete."""
        job = self.status(job_id)
        if job.status != "complete" or not job.model_id:
            raise RuntimeError(
                f"Job {job_id} not complete (status={job.status}). "
                f"Cannot retrieve model_id."
            )
        return job.model_id

    def validate(self, job_id: str, prompt: str, system_prompt: str = "") -> str:
        """
        Run inference with the fine-tuned adapter to verify it works.

        Uses mlx_lm.generate with the adapter loaded on top of base model.
        """
        job = self.status(job_id)
        if not job.succeeded:
            raise RuntimeError(f"Job {job_id} not complete (status={job.status})")

        adapter_path = job.model_id
        return self.inference_with_adapter(prompt, adapter_path, system_prompt)

    # ── Additional helper methods ─────────────────────────────────────

    def launch_from_examples(
        self,
        examples: list[TrainingExample],
        system_prompt: str = "",
        config: dict[str, Any] = None,
    ) -> TrainingJob:
        """
        Convenience method: generate JSONL from TrainingExample list, then launch.

        This is the primary integration point with DataFactory.
        """
        from foundry.factory.data_factory import DataFactory

        output_dir = Path(self.config.output_dir) / "staged"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "train.jsonl"

        factory = DataFactory(pool=None)
        factory.save_training_data(examples, str(jsonl_path), system_prompt)

        return self.launch(str(jsonl_path), config)

    def inference_with_adapter(
        self,
        prompt: str,
        adapter_path: str,
        system_prompt: str = "",
        max_tokens: int = 256,
    ) -> str:
        """
        Run inference using base model + LoRA adapter directly via mlx_lm.

        Useful for quick validation without deploying to Ollama.
        """
        # Escape quotes in prompts for safe embedding in python -c
        safe_prompt = prompt.replace('"', '\\"').replace('\n', '\\n')
        safe_sys = system_prompt.replace('"', '\\"').replace('\n', '\\n')

        script = f'''
from mlx_lm import load, generate

model, tokenizer = load(
    "{self.config.base_model}",
    adapter_path="{adapter_path}",
)

messages = []
if "{safe_sys}":
    messages.append({{"role": "system", "content": "{safe_sys}"}})
messages.append({{"role": "user", "content": "{safe_prompt}"}})

chat_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = generate(model, tokenizer, prompt=chat_prompt, max_tokens={max_tokens})
print(response)
'''
        proc = subprocess.run(
            [self._python, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Inference failed: {proc.stderr[-500:]}")
        return proc.stdout.strip()

    # ── Private helpers ───────────────────────────────────────────────

    def _find_python(self) -> str:
        """Find the conda env python with mlx-lm installed."""
        candidates = [
            os.path.expanduser("~/anaconda3/envs/agent-foundry/bin/python"),
            os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "python"),
            "python",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "python"

    def _prepare_data_from_jsonl(self, jsonl_path: str, data_dir: Path) -> None:
        """Split a JSONL file into train.jsonl and valid.jsonl."""
        with open(jsonl_path) as f:
            lines = f.readlines()

        split_idx = max(1, int(len(lines) * (1 - self.config.val_split)))
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:] if split_idx < len(lines) else lines[:1]

        (data_dir / "train.jsonl").write_text("".join(train_lines))
        (data_dir / "valid.jsonl").write_text("".join(val_lines))
        print(f"[MLXLoRA] Data: {len(train_lines)} train, {len(val_lines)} valid")

    def _parse_output(self, stdout: str) -> tuple[float, float, int]:
        """Parse mlx_lm training output for loss metrics."""
        import re

        train_loss = 0.0
        val_loss = 0.0
        iters = 0

        train_matches = re.findall(
            r'[Ii]ter\s+(\d+).*?[Tt]rain\s+loss\s+([0-9.]+)', stdout
        )
        if train_matches:
            iters = int(train_matches[-1][0])
            train_loss = float(train_matches[-1][1])

        val_matches = re.findall(r'[Vv]al\s+loss\s+([0-9.]+)', stdout)
        if val_matches:
            val_loss = float(val_matches[-1])

        return train_loss, val_loss, iters
