#!/usr/bin/env python3
"""
Minimal demo of the training (private) mode, with random actions.

You get a whole environment of your own and drive all its agents
(``agent_0``, ``agent_1``, ...).  The token and the name are read from two
environment variables:

    export USER_TOKEN=<your token>
    export USER_NAME=<your username>
    python private_mode.py

The "human" render mode needs pygame, and an administrator token: rendering is
reserved to the organizers, in the training mode as in the contest mode.
"""

import os

import pettingzoo

# Importing pzclient registers "csc53439ep/csc53439ep-remote-v1" in the PettingZoo registry
import pzclient  # noqa: F401

env = pettingzoo.make(
    "parallel",
    "csc53439ep/csc53439ep-remote-v1",
    api_url="https://csc53439ep.jdhp.org/training/api",  # the training server
    token=os.environ["USER_TOKEN"],
    render_mode=None,
)

print(f"{os.environ['USER_NAME']} drives {env.possible_agents}")

observations, infos = env.reset(seed=42)

for _ in range(500):  # the episode is eternal: bound the loop
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if not env.agents:  # only if the server sets a maximum episode length
        observations, infos = env.reset()

env.close()
