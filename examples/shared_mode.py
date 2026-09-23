#!/usr/bin/env python3
"""
Minimal demo of the contest (shared) mode, with random actions.

Your agent, named after you, plays in the world shared with the other
participants.  The token and the name are read from two environment variables:

    export USER_TOKEN=<your token>
    export USER_NAME=<your username>
    python shared_mode.py

The "human" render mode needs pygame, and an administrator token: in the
contest mode, the picture of the shared world is reserved to the organizers.
"""

import os

import pettingzoo

# Importing pzclient registers "alife/alife-remote-v1" in the PettingZoo registry
import pzclient  # noqa: F401

agent = os.environ["USER_NAME"]  # in the shared world, your agent is named after you

env = pettingzoo.make(
    "parallel",
    "alife/alife-remote-v1",
    api_url="https://csc53439ep.jdhp.org/api",
    token=os.environ["USER_TOKEN"],
    render_mode="human",
)

observations, infos = env.reset()  # join the shared world

for _ in range(500):  # the episode is eternal: bound the loop
    actions = {agent: env.action_space(agent).sample()}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if not env.agents:  # only if the organizers set a maximum episode length
        observations, infos = env.reset()

env.close()  # leave the shared world
