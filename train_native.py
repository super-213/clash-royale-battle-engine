"""Train the native spatial-heatmap recurrent PPO policy directly."""

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
import torch

if __package__:
    from .native_recurrent_model import (
        NativeWorldModelPPO,
        NativeWorldModelRecurrentPolicy,
    )
    from .native_vec_env import NativeBatchVecEnv
else:
    from native_engine.native_recurrent_model import (
        NativeWorldModelPPO,
        NativeWorldModelRecurrentPolicy,
    )
    from native_engine.native_vec_env import NativeBatchVecEnv


def require_training_device(device):
    """Reject an unavailable accelerator instead of silently using the CPU."""
    requested = str(device).lower()
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "This PyTorch build has no MPS support. Use the clash-royale "
                "Conda environment on the Apple Silicon host."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable in this process. Run the "
                "training command from the normal macOS desktop session; do not "
                "fall back to CPU for the formal M5 Pro training run."
            )
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError("device must be 'mps' or 'cpu'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=26789)
    parser.add_argument("--battles", type=int, default=16)
    parser.add_argument("--decision-ticks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=8192)
    parser.add_argument(
        "--device",
        default="mps",
        choices=("mps", "cpu"),
        help="Neural-network device. Defaults to the M5 Pro GPU through MPS.",
    )
    parser.add_argument(
        "--output", default="artifacts/native_spatial_heatmap_smoke"
    )
    parser.add_argument("--tensorboard-log", default=None)
    args = parser.parse_args()
    if 8192 % args.battles:
        raise ValueError("battles must divide the 8,192-decision PPO rollout")
    if args.total_timesteps <= 0:
        raise ValueError("total-timesteps must be positive")
    device = require_training_device(args.device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = VecMonitor(NativeBatchVecEnv(
        battles=args.battles,
        port=args.port,
        decision_ticks=args.decision_ticks,
        seed=args.seed,
        learner_roles="alternating",
    ))
    rollout_steps = 8192 // args.battles
    checkpoint = CheckpointCallback(
        save_freq=max(rollout_steps, 1),
        save_path=str(output.parent),
        name_prefix=output.name,
    )
    model = NativeWorldModelPPO(
        NativeWorldModelRecurrentPolicy,
        env,
        n_steps=rollout_steps,
        batch_size=512,
        n_epochs=4,
        tbptt_steps=32,
        auxiliary_coef=0.1,
        auxiliary_horizon=4,
        gamma=0.997,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=1.0,
        clip_range=0.2,
        device=device,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint,
            progress_bar=False,
        )
        model.save(output)
    finally:
        env.close()

    print(
        f"saved={output.with_suffix('.zip')} timesteps={model.num_timesteps} "
        f"device={device} "
        f"schema=native_model_v2 backend=native_batch_v1"
    )


if __name__ == "__main__":
    main()
