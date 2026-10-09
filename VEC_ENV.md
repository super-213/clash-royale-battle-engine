# Native Batch VecEnv

`NativeBatchVecEnv` is the reinforcement-learning boundary for the resident
native engine. One instance owns one `Batch`, normally with all 16 slots. Every
`step_async`/`step_wait` pair submits all learner and opponent actions in one
binary `Batch.step` request. It does not create 16 host sockets or call 16
single-battle environments sequentially.

## SB3 contract

The action space is `MultiDiscrete([5, 32, 18])` per slot:

- slot `0` is wait;
- slots `1..4` select the current four-card hand;
- `y, x` select an egocentric arena tile;
- player 1 coordinates are rotated into native world coordinates;
- `action_mask[slot, y, x]` is the conditional legality mask.

Deployment rules are versioned as `standard_1v1_dynamic_v1`. Troops can use the
full legal home territory, including rear and outer cells, excluding live crown
tower footprints and arena/river boundaries. Destroying an enemy princess tower
unlocks its lane's forward territory and bridge cells; destroying both unlocks
the union. Each episode records the initial crown tower identities, so a tower
missing from the initial capture does not falsely unlock territory. The same
mask is used for learner actions, opponents and `env_method("action_masks")`.
Fireball and Arrows can target all 576 arena cells when affordable. With all six
crown towers alive, troops have 222 cells; one destroyed enemy princess tower
adds 37 cells, and two add 74. A destroyed own princess tower also frees its
9-cell footprint.

These boundaries were checked against actual native card-play events and troop
spawn positions on the supported 15.535.13 engine, for both player roles and
both lanes. The observation/action tensor shapes stay unchanged, so existing
checkpoints can load; further training is needed to learn the newly available
placements. Fixed evaluation results before and after this rule change should
be treated as different environment versions. Training summaries record
`deployment_rules_version`.

When a battle finishes, its final observation is placed in
`info["terminal_observation"]`. Only those terminal slots are recreated through
`Batch.reset_slots`; surviving slots preserve their native manager, generation,
tick and episode history. The observation returned for a finished slot is the
new tick-90 episode, matching the Stable-Baselines3 VecEnv auto-reset contract.

The adapter supports `VecMonitor` and the ordinary SB3 `VecEnv` methods used by
training code. `env_method("action_masks")` returns one mask per selected slot.
When the opponent is an SB3-style model, all 16 opponent observations are sent
through one batched `predict` call rather than 16 sequential model forwards.

## Model-facing observation (`native_model_v2`)

| Key | Shape | Intended encoder |
|---|---:|---|
| `spatial` | `16 x 32 x 18` | CNN arena encoder |
| `entities` | `128 x 24` | entity Transformer |
| `entity_mask` | `128` | Transformer padding mask |
| `hand` | `5` | card embeddings (hand + next card) |
| `scalar` | `12` | current public context |
| `public_belief` | `20` | explicit opponent hand/next-card cycle belief |
| `action_history` | `32 x 8` | GRU/LSTM/temporal Transformer |
| `temporal` | `20` | Per-decision scene delta for persistent memory |
| `action_mask` | `5 x 32 x 18` | masked hierarchical action head |

Opponent elixir in `scalar` is reconstructed from the public play ledger and
regeneration schedule. The opponent's native hidden player record is not used.
Successful public plays, not wait decisions or rejected attempts, populate the
temporal history. `public_belief` enumerates cycle states consistent with public
plays; after four opponent plays its hand and next-card belief is exact. A schema
change must use a new schema name and new checkpoints. Version 2 adds continuous
deltas for tower HP, elixir, entity spawn/death, combat HP, target state, crowns
and public plays.

## Usage

```python
from native_engine import NativeBatchVecEnv
from stable_baselines3.common.vec_env import VecMonitor

env = VecMonitor(NativeBatchVecEnv(battles=16, port=26789, seed=1))
observations = env.reset()
# model actions have shape (16, 3)
observations, rewards, dones, infos = env.step(actions)
env.close()
```

Run deterministic adapter tests:

```sh
conda run -n clash-royale python -m unittest discover -s tests -p 'test_*.py' -v
```

After the emulator lane is ready, run a bounded real-engine smoke:

```sh
conda run -n clash-royale python vec_env_smoke.py \
  --port 26789 --battles 16 --steps 256 --ticks 10
```

The smoke intentionally does not call `stop_resident`; it closes the batch
socket and leaves the lane available for another batch client.

## Recurrent policy with an auxiliary world model

`native_spatial_model.py`, `native_cross_modal.py` and
`native_recurrent_model.py` contain the default trainable policy for this schema:

```text
entities/mask -- entity Transformer ----------------------+
       ^                 ^                                |
history --query entities-+--> event LSTM -----------------+
belief ---query entities/history -------------------------+--> fusion
hand -----query entities/history/belief ------------------+      |
spatial -- full-resolution CNN --> 32x18 spatial tokens           |
temporal/scalar ------------------------------------------+      v
                                                   persistent LSTMs
                                                          |
card queries x 32x18 spatial tokens <--- actor context ----+
              |
       masked P(slot) * P(tile | slot, card, context)
```

The event LSTM summarizes the explicit public-play window after its tokens have
queried current entities. Card tokens query entities, history and public belief.
The CNN never compresses the actor's map to one vector: it preserves all 576
arena positions at native resolution. The actor conditions every card token on
persistent recurrent context, then uses it as a query over the full spatial map,
producing four separate `32 x 18` placement heatmaps. The hierarchical
distribution samples the card first and only then samples from that card's
masked heatmap. A second, stateful actor/critic LSTM carries hidden state between
decisions.
RecurrentPPO resets only slots whose `episode_start` is true. Training uses
padded sequence minibatches with hard TBPTT boundaries, 32 decisions by default.
Neither LSTM is treated as a literal predictor of future states; expected future
outcome is learned by PPO's critic. During training, a separate auxiliary head
also predicts next-transition dynamics and a short-horizon scene value from the
same encoder and recurrent latent. Entity labels are generated by the environment
from adjacent native states; other labels are constructed inside sequence
minibatches. Neither kind enters inference observations.

The auxiliary tasks are next tower-HP delta, entity death, entity position delta,
five-way public damage-source class, opponent public-elixir interval, possible
opponent hand, possible next card, and future tower/crown scene value. Entity
death and movement labels use exact native IDs, while the public 24-field entity
token continues to omit those IDs. `step_wait` emits training-only
`info["entity_target_schema"] = "native_entity_id_v1"` and
`info["entity_targets"]`, a float32 `(128, 4)` array of `[death, dx, dy, valid]`.
Rows follow the current observation's entity ordering. Survivors are looked up
in the complete next native state, so observation truncation cannot imply death.
Movement uses clipped egocentric normalized coordinates, including p1 rotation;
buildings and padding have `valid=0`, and dead entities have no position loss.
Terminal labels are generated before automatic reset and cannot match reused IDs
in a new episode.

`NativeWorldModelPPO` stores these labels in a separate recurrent rollout-buffer
array. Sampling applies the same environment ordering, random split, sequence
boundaries and padding as policy observations. Entity losses retain the adjacent
pair mask, excluding episode/TBPTT boundaries and padding. An environment without
the versioned labels fails explicitly rather than falling back to approximate
matching. The observation schema, network parameters and inference API are
unchanged; existing v5 checkpoints can resume training with the new labels.

The default v6 `NativeActionConditionedWorldModelPPO` /
`NativeActionConditionedWorldModelRecurrentPolicy` additionally condition dynamics
on the learner's executed card and placement. `info["executed_action_schema"]` is
`executed_card_xy_v1`, and `info["executed_action"]` is float32 `(11,)`:
`[wait, eight card-identity one-hot entries, x/18000, y/32000]` in learner coordinates.
The environment requires both a successful native receipt and a matching public
play event. Identity comes from the receipt and coordinates from the submitted
placement, before hand cycling or automatic reset can change the input. Waiting,
masked actions, rejected actions and missing play events all use `[1, 0, ..., 0]`.

`ActionTargetRecurrentDictRolloutBuffer` stores these execution features separately
from observations and PPO's sampled actions. They follow the identical shuffle,
sequence boundaries and padding as the state at t. Tower delta, damage source,
entity death/movement and future scene value use fused state/action features;
current opponent elixir/hand/next-card heads remain state-only. Future value is
conditioned on this first action; later actions and opponent responses are drawn
from the rollout policies. This is auxiliary training, not planning or a full
counterfactual simulator. Inference action masks and actor/critic inputs are
unchanged. Diagnostic `policy.auxiliary_predictions(executed_actions)` accepts a
batch of these features after evaluating the policy on the corresponding states.
v5 classes retain their original state-only head and can load/resume v5 models;
v6 is a separate checkpoint architecture and does not automatically convert v5.
Training summaries record the architecture and executed-action schema.

The default v6 policy has 1,824,036 trainable parameters; v5 has 1,789,092.
Unit coverage verifies per-slot hidden-state reset, finite recurrent PPO updates,
hard TBPTT sequence bounds, legal cross-modal actions, modality-sensitive tokens,
and non-zero gradients through hand/history attention, the full-resolution CNN
and card/spatial projections. A real M5 Pro MPS process also completed
rollout/backward, checkpoint save/reload and post-load masked inference with
finite `(2, 4, 32, 18)` heatmaps. The exact parameter count for each custom
configuration is written into its `summary.json` and `summary.md`.

Run formal recurrent training through the profile-driven launcher after the
emulator lane passes `vec_env_smoke`:

```sh
./train.sh
```

With 16 battles, PPO uses 512 vector steps per rollout, exactly 8,192 learner
decisions. The default command is an engineering smoke, not evidence that the
policy is strong. `train_native.py` defaults to `mps` and now fails immediately
if MPS is unavailable; it will not silently move a formal M5 Pro run to CPU.

## Verified native smoke (2026-09-28)

The complete path has run against the supported real APK and probe:

- single-battle fidelity: `result=1`, crowns `(0, 1)`, final tick `3835`;
- VecEnv: 16 slots, 512 vector steps, 8,192 decisions, 15 terminal slot resets;
- headless SwiftShader AVD: about 706 decisions/s and 7,053 native ticks/s;
- PPO: one 8,192-timestep rollout plus four MPS update epochs completed;
- saved checkpoint reloaded with the hybrid policy and extractor;
- post-load `mps:0` real-engine action sample had zero mask violations, finite
  parameters, and advanced all checked slots from tick 90 to tick 100.

The M5 Pro MPS rollout reported about 365 decisions/s, versus about 202 decisions/s
in the earlier CPU engineering run. MPS validation exposed and fixed PyTorch's
non-divisible `AdaptiveAvgPool2d` limitation; the CNN now uses an equivalent
MPS-supported reduction. The reloadable smoke checkpoint is
`/tmp/native_hybrid_mps_smoke_v2.zip` (SHA-256 `83e5da…bd34`). These are engineering
results, not a policy-strength claim or a replacement for a longer warm benchmark.
