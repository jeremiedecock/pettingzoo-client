#!/usr/bin/env python3
"""
Minimal demo of the training (private) mode, with random actions.

You get a whole environment of your own and drive all its agents
(``agent_0``, ``agent_1``, ...).  The token and the name are read from two
environment variables:

    export USER_TOKEN=<your token>
    export USER_NAME=<your username>
    python private_mode.py

The "human" render mode needs pygame.
"""

import os

import pettingzoo

# Importing pzclient registers "alife/alife-remote-v1" in the PettingZoo registry
import pzclient  # noqa: F401

env = pettingzoo.make(
    "parallel",
    "alife/alife-remote-v1",
    api_url="http://localhost:8000/api",  # the URL of the training server
    token=os.environ["USER_TOKEN"],
    render_mode="human",
)

print(f"{os.environ['USER_NAME']} drives {env.possible_agents}")

observations, infos = env.reset(seed=42)

for _ in range(500):  # the episode is eternal: bound the loop
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if not env.agents:  # only if the server sets a maximum episode length
        observations, infos = env.reset()

env.close()
