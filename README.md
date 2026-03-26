---
title: OpenEnv Support Triage
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Support Triage OpenEnv

I built this OpenEnv environment around a real support workflow: triaging incoming customer tickets.

The agent has to do the same things a support engineer does under pressure — set priority, route to the correct team, escalate incidents, and write safe customer replies. My goal was to keep this practical and operations-focused.

## Why this environment

Support triage is one of the most common day-to-day tasks in SaaS teams. In this environment, I expect an agent to:

- identify urgent incidents quickly,
- route tickets to the right teams,
- avoid unsafe or premature resolution,
- communicate clearly to customers.

I tried to capture those constraints with deterministic graders and shaped rewards over the full episode.

## OpenEnv interface

The environment implements the standard OpenEnv API:

- `reset(task_id)` → returns typed `Observation`
- `step(action)` → returns typed `observation`, `reward`, `done`, `info`
- `state()` → returns internal environment state snapshot

Typed models are in `app/models.py`:

- `Observation`
- `Action`
- `Reward`

Metadata is provided in `openenv.yaml`.

## Action space

The action schema is defined in Pydantic (`Action`) and also exposed through `/tasks`.

`action_type` options:

- `set_priority` (`ticket_id`, `priority`)
- `assign_team` (`ticket_id`, `team`)
- `add_label` (`ticket_id`, `label`)
- `draft_reply` (`ticket_id`, `message`)
- `resolve_ticket` (`ticket_id`)
- `escalate_ticket` (`ticket_id`)
- `noop`

## Observation space

Each observation contains:

- task metadata (`task_id`, `task_name`, `instruction`)
- episode progress (`step_count`, `max_steps`)
- current queue of typed `Ticket` objects
- recent environment events

## Tasks (easy → medium → hard)

1. **easy** — one critical outage ticket in a small queue
   - Goal: prioritize correctly, route to engineering, respond safely.
2. **medium** — mixed queue (billing + security + feature request)
   - Goal: make good routing/severity decisions across different ticket types.
3. **hard** — SLA-critical multi-incident queue
   - Goal: balance outage + security urgency while still handling refund/bug requests responsibly.

All tasks have deterministic graders returning scores in `[0.0, 1.0]`.

## Reward design

Reward is shaped across the trajectory (not just final success/fail):

- positive signal for score improvement (`progress_delta`)
- small per-step cost to discourage wasteful behavior
- penalties for invalid actions
- penalties for repeated loop-like actions
- safety penalty for resolving tickets without drafting a customer response

This gives the agent useful feedback during an episode, not only at the end.

## API endpoints

- `POST /reset` — reset task (`{"task_id":"easy|medium|hard"}`)
- `POST /step` — apply one action
- `GET /state` — inspect current state
- `GET /tasks` — list tasks + action schema
- `GET /grader` — current grader score
- `POST /baseline` — run baseline policy on all three tasks

## Local setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t support-triage-openenv .
docker run -p 7860:7860 support-triage-openenv
```

### Quick local API smoke test (curl)

After the container is running:

```bash
curl -s http://localhost:7860/tasks
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{\"task_id\":\"easy\"}"
curl -s -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{\"action_type\":\"set_priority\",\"ticket_id\":\"T-EASY-1\",\"priority\":\"critical\"}"
curl -s http://localhost:7860/grader
curl -s -X POST http://localhost:7860/baseline -H "Content-Type: application/json" -d "{\"model\":\"gpt-4o-mini\",\"max_steps\":12}"
```

## Baseline inference script

`baseline.py` runs easy/medium/hard and prints reproducible scores.

- Uses OpenAI Python client when `OPENAI_API_KEY` is set (`temperature=0`).
- Falls back to deterministic heuristic policy if the key is missing or API call fails.

Run:

```bash
python baseline.py --model gpt-4o-mini
```

## Hugging Face Spaces deployment

I deployed this on a **Docker** Space. Add the `openenv` tag in Space settings.

Container starts on port `7860` and serves API from `app.server:app`.

Live Space URL:

- https://vinayak-1409-openenv-support-triage.hf.space

## Validation checklist

Before submitting:

1. `openenv validate` (if installed in your environment)
2. `docker build` and `docker run` succeed
3. `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline` respond correctly
4. baseline script completes and returns scores for easy/medium/hard

## Baseline reference scores

Current reference output (`python baseline.py`):

- easy: `0.85`
- medium: `0.55`
- hard: `0.55`
- average: `0.65`

## Submission links

- GitHub: https://github.com/vinayak-05/OpenEnv
- Hugging Face Space: https://huggingface.co/spaces/Vinayak-1409/openenv-support-triage
