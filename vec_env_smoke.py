"""Short real-engine smoke for the 16-slot SB3 VecEnv.

Run as ``python -m native_engine.vec_env_smoke --port 26789`` after bootstrapping
the supported APK and installing the resident-reset probe.
"""

import argparse
import time

import numpy as np

if __package__:
    from .native_vec_env import NativeBatchVecEnv
else:
    from native_engine.native_vec_env import NativeBatchVecEnv


def legal_actions(observation, randomizer):
    actions = np.zeros((observation["action_mask"].shape[0], 3), dtype=np.int64)
    for slot, mask in enumerate(observation["action_mask"]):
        legal = np.argwhere(mask[1:] > 0)
        if len(legal) and randomizer.random() >= 0.2:
            card, y, x = legal[randomizer.integers(len(legal))]
            actions[slot] = (int(card) + 1, int(y), int(x))
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=26789)
    parser.add_argument("--battles", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    randomizer = np.random.default_rng(args.seed)
    env = NativeBatchVecEnv(
        battles=args.battles,
        port=args.port,
        decision_ticks=args.ticks,
        seed=args.seed,
    )
    completed = simulated_ticks = 0
    try:
        observation = env.reset()
        previous_ticks = [state.tick for state in env.env_method("get_state")]
        started = time.perf_counter()
        for _ in range(args.steps):
            observation, _, dones, infos = env.step(
                legal_actions(observation, randomizer)
            )
            for slot, info in enumerate(infos):
                simulated_ticks += max(0, int(info["tick"]) - previous_ticks[slot])
                previous_ticks[slot] = env.slots[slot].state.tick
            completed += int(np.count_nonzero(dones))
        seconds = time.perf_counter() - started
    finally:
        env.close()

    decisions = args.steps * args.battles
    print(
        f"battles={args.battles} vector_steps={args.steps} "
        f"decisions={decisions} completed={completed}"
    )
    print(
        f"elapsed={seconds:.3f}s decisions_per_second={decisions / seconds:.1f} "
        f"ticks_per_second={simulated_ticks / seconds:.0f}"
    )


if __name__ == "__main__":
    main()
