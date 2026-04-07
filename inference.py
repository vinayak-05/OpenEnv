from __future__ import annotations

import argparse
import os

from app.environment import SupportTriageEnv
from app.policy import heuristic_action, llm_action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _run_task(env: SupportTriageEnv, task_id: str, model: str, max_steps_override: int | None) -> None:
    observation = env.reset(task_id=task_id)
    done = False
    step = 0
    max_steps = max_steps_override or observation.max_steps

    print(f"[START] task={task_id}", flush=True)

    while not done and step < max_steps:
        try:
            action = llm_action(
                observation,
                model=model,
                api_base_url=API_BASE_URL,
                api_key=HF_TOKEN,
            )
        except Exception:
            action = heuristic_action(observation)

        result = env.step(action)
        observation = result.observation
        step += 1
        done = result.done
        print(f"[STEP] step={step} reward={result.reward.value:.4f}", flush=True)

    final_score = env.grade()
    print(f"[END] task={task_id} score={final_score:.4f} steps={step}", flush=True)


def run(model: str = MODEL_NAME, max_steps: int | None = None) -> None:
    env = SupportTriageEnv()
    for task_id in ["easy", "medium", "hard"]:
        _run_task(env=env, task_id=task_id, model=model, max_steps_override=max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with structured stdout blocks")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    run(model=args.model, max_steps=args.max_steps)


def main_entry() -> None:
    main()


if __name__ == "__main__":
    main()
