# Loba AI

> A card-game AI that grew out of family table nights.

La Loba is a rummy-style card game we grew up playing. After one too many rounds at the family table arguing over whether you *should* have discarded that seven of hearts, my brother and I made a deal: instead of arguing, we'd each train a bot, plug them into a server, and let the bots fight it out. This repo is my side of that bet — the engine, the training pipeline, a playground to watch models play each other, and the protocol bridges that let his bot and mine sit at the same virtual table.

It started as a weekend project and turned into a real piece of code. We're putting it out as open source in case it's useful — or fun — for anyone else.

## What's in here

- **Game engine** — full La Loba rules: piernas, escaleras, jokers, reenganches, match-to-100 scoring (`loba_ai.engine`, `loba_ai.melds`, `loba_ai.rules`).
- **Gymnasium environments** — single-round, full-match, and the "remote-like" variants the trained models actually use (`loba_ai.env`, `loba_ai.match_env`, `loba_ai.remote_like_*`).
- **Heuristic bots** — a simple priority-based agent and a stronger scored-action agent. Useful as opponents for training and as sanity baselines (`loba_ai.agents`).
- **Training pipeline** — Behavior Cloning to bootstrap from heuristic play, then PPO fine-tuning with `sb3-contrib`'s `MaskablePPO`. With tactical reward shaping for project preservation, unsafe-meld penalties, and snapshot-based self-play (`loba_ai.training`).
- **FastAPI playground + arena** — a small web UI where you can register `.zip` models, create matches, and watch model-vs-model autoplay or play yourself (`loba_ai.api`, `web/`).
- **Remote bridge** — a WebSocket client that connects this backend to an external La Loba server, so two independently-trained bots can play each other across machines (`loba_ai.remote`).

## Quickstart

```bash
git clone <your-fork-url> loba-ai
cd loba-ai
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

Python 3.11+. Dependencies: numpy, gymnasium, torch, stable-baselines3, sb3-contrib, fastapi, uvicorn, websockets, httpx.

## The three play modes

### 1. Playground — local UI

The simplest way to see the bots in action. Start the server:

```bash
uvicorn loba_ai.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, register your `.zip` model files, and create a match. You can mix and match player kinds:

- `model` — a trained policy loaded from a checkpoint
- `heuristic` — the rule-based bot
- `random` — sanity-check baseline
- `human` — you, clicking buttons in the browser

`Autoplay` runs bot-vs-bot games turn by turn so you can watch the policies actually decide.

### 2. Arena — room-based duels via API

Arena is the "two brothers, two laptops, one network" mode. The host creates a room, the guest joins with a token, and each agent plays its seat through HTTP. The server filters state per-token so neither side can see the other's hand.

```bash
# Host creates a room
curl -X POST http://192.168.x.x:8000/arena/rooms \
  -H "Content-Type: application/json" \
  -d '{"host_model_name":"mi-bot","target_points":100}'

# Guest joins
curl -X POST http://192.168.x.x:8000/arena/rooms/{room_id}/join \
  -H "Content-Type: application/json" \
  -d '{"guest_name":"hermano-bot"}'
```

The full reference — endpoints, action encoding, state shape, error codes — lives in `arena_api.md`. A bot needs only `get_state`, `choose_action`, `play_action` to participate.

### 3. Remote — WebSocket bridge to an external server

`/remote` lets this backend act as a *client* to someone else's La Loba server (the protocol my brother runs). The adapter handles registration, observation/action mapping, and reconnection.

```bash
# UI for managing remote sessions
open http://localhost:8000/remote

# Or via API:
curl -X POST http://localhost:8000/remote/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "url": "ws://brother-host:8765/",
    "name": "my-bot",
    "model_name": "mi-bot-local"
  }'
```

The wire format is documented in `remote_api.md` (our side) and was originally specified for cross-language compatibility — any client that speaks WebSockets and JSON can plug in.

## Training a model

Models are trained in three stages: collect a heuristic dataset, behavior-clone it, then RL fine-tune from there.

### 1. Collect a heuristic dataset

Have the heuristic bot play matches and dump `(observation, mask, action)` tuples:

```bash
python -m loba_ai.training.collect_heuristic_data \
  --num-players 3 \
  --opponent-mix heuristic,strong_heuristic,mixed_heuristic \
  --seat-0-policy strong_heuristic \
  --target-samples 1000000 \
  --skip-trivial \
  --target-points 100 \
  --max-reenganches 2 \
  --output artifacts/bc_dataset.npz
```

`--skip-trivial` drops decisions where only one action is legal (no learning signal). `--opponent-mix` keeps the state distribution broad to reduce covariate shift downstream.

### 2. Behavior clone

Supervised pretraining so PPO doesn't start from a near-random policy:

```bash
python -m loba_ai.training.pretrain_bc \
  --dataset artifacts/bc_dataset.npz \
  --num-players 3 \
  --policy-net-arch 512,512 \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --output artifacts/bc_policy.zip
```

### 3. RL fine-tune

`MaskablePPO` over a fixed action space of 256 actions (only the legal subset is masked in at each step). Reward combines a match-level signal (win bonus / loss penalty), round-level shaping (encourage melds, penalize discarding into projects), and tactical signals (unsafe-meld penalty, project-break penalty, etc.):

```bash
python -m loba_ai.training.train_remote_like_match_smart_rl \
  --resume-from artifacts/bc_policy.zip \
  --num-players 3 --opponent heuristic \
  --policy-net-arch 512,512 \
  --timesteps 5000000 \
  --n-envs 40 --vec-env subproc \
  --n-steps 256 --batch-size 4096 \
  --ent-coef 0.0005 --learning-rate 2.5e-5 \
  --match-win-bonus 200 \
  --match-loss-penalty 100 \
  --reward-unsafe-meld-penalty 5 \
  --reward-discard-project-penalty 0.5 \
  --reward-discard-break-project-penalty 0.75 \
  --eval-freq 50000 --eval-episodes 50 \
  --checkpoint-freq 500000 \
  --tensorboard-log artifacts/tensorboard \
  --model-out artifacts/my_model
```

The trainer also supports snapshot-based self-play (`--self-play-snapshot-dir`, `--self-play-snapshot-freq`, `--self-play-pool-size`) for later stages once a heuristic-beating baseline exists.

### 4. Evaluate

Run a fixed-seed bake-off across checkpoints:

```bash
python -m loba_ai.training.evaluate_remote_like_match_checkpoints \
  --models artifacts/my_model.zip artifacts/my_model_best/best_model.zip \
  --episodes 300 \
  --num-players 3 \
  --opponent heuristic \
  --seed 1300 \
  --json-out artifacts/eval.json
```

Models are ranked by `win_rate`, then `mean_reward`, then `agent_elimination_rate`, then `avg_match_score`.

### What the model sees

The observation is a flat vector built by `build_smart_match_obs`. It includes:

- the agent's own hand (card-by-card encoding)
- the top of the discard pile and recent discard history
- opponent hand sizes and elimination flags
- current phase, action history, match scores, reenganches used
- a preview of the legal actions plus per-action tactical features (`is_project_card`, `breaks_project`, `post_action_unsafe_excess`, `meld_points_removed`, etc.)

Action encoding is fixed-width (256 slots). Per-action tactical features sit alongside the action mask, which gives the policy a much easier handle on "is this a smart move?" than expecting it to derive everything from the global hand state.

### What the rewards say

Match-level: `match_win_bonus` and `match_loss_penalty` define the long-horizon target. Round-level: bonuses for laying piernas/escaleras, extending melds, and "cruzar"; penalties for discarding when a productive play was available, or for breaking your own projects. Tactical: a penalty for laying a meld that leaves you stranded above the safe zone (`reward_unsafe_meld_penalty`), a penalty for discarding a project card when you have a higher dead alternative, and a small bonus for the inverse — clearing the highest dead point while preserving projects.

The full shaping reference is in `loba_ai/training/train_remote_like_match_smart_rl.py`.

## Tests

```bash
pytest
```

Coverage includes the engine, melds, environment dynamics, the playground API, and the training script. CI-friendly: no GPU required, no external network calls.

## Project layout

```
loba_ai/
  cards.py, engine.py, melds.py, rules.py    # game core
  env.py, match_env.py, smart_match_env.py   # gymnasium envs (round-level → match-level)
  remote_like_*.py                            # match-style envs used by the current models
  agents/                                     # heuristic + random bots
  remote/                                     # WebSocket client for /remote
  playground/                                 # API service for the local UI
  training/                                   # data collection, BC, PPO, evaluation, ablation
  api.py                                      # FastAPI entry point
web/                                          # static UI (playground, arena, remote)
tests/                                        # pytest suite
arena_api.md, remote_api.md                   # protocol references
```

## Acknowledgements

This project ships with playing-card SVGs from David Bellot and Huub de Beer's [SVG-cards](https://github.com/htdebeer/SVG-cards), released under LGPL-2.1. Full attribution and the license text are in `THIRD_PARTY_LICENSES.md` and `web/assets/cards/SVG-cards-LICENSE-LGPL-2.1.txt`.

## License

MIT — see `LICENSE`.

---

If you do something fun with this — train a stronger bot, add another card game on top, port the protocol to your language of choice — open an issue or send a PR. We'd love to see it.
