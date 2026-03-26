from __future__ import annotations

import argparse
import json
from typing import Dict

from app.environment import SupportTriageEnv
from app.policy import heuristic_action, llm_action


def run_baseline(model: str = "gpt-4o-mini", max_steps_override: int | None = None) -> Dict[str, float]:
    env = SupportTriageEnv()
    scores: Dict[str, float] = {}

    for task_id in ["easy", "medium", "hard"]:
        observation = env.reset(task_id=task_id)
        done = False
        steps = 0
        max_steps = max_steps_override or observation.max_steps

        while not done and steps < max_steps:
            try:
                action = llm_action(observation, model=model)
            except Exception:
                action = heuristic_action(observation)

            result = env.step(action)
            observation = result.observation
            done = result.done
            steps += 1

        scores[task_id] = env.grade()

    scores["average"] = round(sum(scores.values()) / 3.0, 4)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible baseline across all tasks")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    scores = run_baseline(model=args.model, max_steps_override=args.max_steps)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
