"""Drive an OASIS Reddit simulation directly, biased toward content creation.

This is a custom thin wrapper around `oasis` (the multi-agent simulator that
backs MiroFish). It mirrors the MiroFish CLI runner but:

  - Restricts the LLM-action space to {create_post, create_comment,
    like_post, like_comment, repost} so agents do not waste rounds on
    `refresh`, `trend`, `search_*`, `follow`, `mute`, or `do_nothing`.
  - Activates a configurable proportion of all agents every round (instead of
    a small subsample driven by the time-of-day multiplier).
  - Writes a single SQLite database `reddit_simulation.db` next to the config,
    matching the layout that `extract_oasis_posts.py` expects.

Usage:
    python code/synth/run_oasis.py --config <run_dir>/simulation_config.json --rounds 24
"""

from __future__ import annotations
from pathlib import Path
import argparse
import asyncio
import logging
import os
import random
import sys
import time

# OASIS lives in the MiroFish backend's .venv site-packages.
VENV_SITE = (Path(r"C:\Users\User\iCloudDrive\Unimelb\COMP90090\external\MiroFish\backend\.venv")
             / "Lib" / "site-packages")
sys.path.insert(0, str(VENV_SITE))

# Silence the OASIS UnicodeEncodeError-noisy file handlers; their failures are
# non-fatal but pollute stdout.
class _DropUnicode(logging.Filter):
    def filter(self, record):
        try:
            record.getMessage().encode("utf-8", errors="strict")
            return True
        except UnicodeEncodeError:
            return False

for name in ("social.agent", "social.twitter", "social.rec", "oasis.env", "table"):
    logging.getLogger(name).addFilter(_DropUnicode())

import json
import oasis
from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
from camel.models import ModelFactory
from camel.types import ModelPlatformType


CONTENT_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_POST,
    ActionType.LIKE_COMMENT,
    ActionType.REPOST,
]
POSTS_ONLY_ACTIONS = [ActionType.CREATE_POST]


def build_model():
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", "ollama"))
    if os.environ.get("LLM_BASE_URL"):
        os.environ["OPENAI_API_BASE_URL"] = os.environ["LLM_BASE_URL"]
    model_name = os.environ.get("LLM_MODEL_NAME", "qwen2.5:14b-instruct-q4_K_M")
    print(f"LLM: model={model_name} base_url={os.environ.get('OPENAI_API_BASE_URL', '(default)')}")
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_name,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--activation_rate", type=float, default=0.5,
                        help="Fraction of agents activated per round (0.5 = half).")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--actions", choices=["content", "posts_only"], default="content",
                        help="Action space. 'posts_only' restricts every LLM-action to "
                             "CREATE_POST, maximising post yield.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    sim_dir = cfg_path.parent
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    profile_path = sim_dir / "reddit_profiles.json"

    rng = random.Random(args.seed)
    model = build_model()
    print(f"loading profiles from {profile_path}")
    action_set = POSTS_ONLY_ACTIONS if args.actions == "posts_only" else CONTENT_ACTIONS
    print(f"action set: {[a.value for a in action_set]}")
    agent_graph = await generate_reddit_agent_graph(
        profile_path=str(profile_path),
        model=model,
        available_actions=action_set,
    )
    print(f"agents in graph: {len(list(agent_graph.get_agents()))}")

    db_path = sim_dir / "reddit_simulation.db"
    if db_path.exists():
        db_path.unlink()
        print(f"removed existing DB: {db_path}")

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=str(db_path),
        semaphore=20,
    )
    await env.reset()
    print("env initialised\n")

    # Seed the platform with the initial posts.
    initial_posts = cfg.get("event_config", {}).get("initial_posts", [])
    initial_actions = {}
    for post in initial_posts:
        agent_id = post.get("poster_agent_id")
        content = post.get("content", "")
        try:
            agent = env.agent_graph.get_agent(agent_id)
            existing = initial_actions.get(agent)
            ma = ManualAction(action_type=ActionType.CREATE_POST,
                              action_args={"content": content})
            if existing is None:
                initial_actions[agent] = ma
            else:
                if not isinstance(existing, list):
                    initial_actions[agent] = [existing]
                initial_actions[agent].append(ma)
        except Exception as e:
            print(f"warn: could not seed post for agent {agent_id}: {e}")
    if initial_actions:
        await env.step(initial_actions)
        print(f"seeded {len(initial_posts)} initial posts across {len(initial_actions)} agents")

    # All agents available for round activation.
    all_agents = [a for _, a in env.agent_graph.get_agents()]
    activations_per_round = max(1, int(args.activation_rate * len(all_agents)))
    t0 = time.time()

    for r in range(args.rounds):
        active = rng.sample(all_agents, k=min(activations_per_round, len(all_agents)))
        actions = {a: LLMAction() for a in active}
        await env.step(actions)
        elapsed = time.time() - t0
        if (r + 1) % 2 == 0 or r == 0:
            print(f"  round {r+1}/{args.rounds} | {len(active)} agents active | "
                  f"elapsed {elapsed:.0f}s")

    print(f"\nsimulation complete in {time.time()-t0:.0f}s; db = {db_path}")
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
