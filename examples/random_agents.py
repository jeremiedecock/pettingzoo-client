#!/usr/bin/env python3
"""
Play the world of the MARL contest with random agents.

This script is the starting point given to the participants of the contest: it
connects to the shared world served by `pettingzoo-server`, plays it
with a random policy and reports what happened.  Replace `policy` by your own
agent and you are done.

The client library makes the remote environment a plain PettingZoo environment,
so the loop below is the usual PettingZoo parallel loop::

    import pettingzoo
    import pzclient  # registers "csc53439ep/csc53439ep-remote-v1"

    env = pettingzoo.make("parallel", "csc53439ep/csc53439ep-remote-v1", token="...")

    observations, infos = env.reset()

    while env.agents:
        actions = {agent: policy(observations[agent], env.action_space(agent))
                   for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()

Two things are worth knowing about this environment:

- **the episode is eternal**: no agent ever leaves the episode, so
  ``env.agents`` never becomes empty and ``while env.agents:`` never ends.
  Bound your loop, as ``--steps`` does here.
- **the world is shared**: the other participants play the same world at the
  same time, and each of you controls a single agent (``env.agents`` holds your
  agent alone).  The world has its own clock: a step is played at the end of
  each window of the step timeout of the server, or as soon as every connected
  participant has sent its action.  An agent that thinks for too long plays a
  default action during that step (``infos[agent]["shared_world"]`` tells whether
  its action was played).

Usage
-----
::

    python random_agents.py --token token_abc123 \
                            --api-url http://localhost:8000/api \
                            --steps 500

``PETTINGZOO_API_URL`` and ``PETTINGZOO_TOKEN`` are the environment variable
equivalents of ``--api-url`` and ``--token``.  Use the token given to you by the
organizers of the contest, and the URL of the server to play on:

- ``https://csc53439ep.jdhp.org/api``, the contest (shared) world;
- ``https://csc53439ep.jdhp.org/training/api``, a training world of your own,
  where you drive all the agents.

``--render-mode human`` shows the world in a window refreshed at every step
(pygame must be installed).  Rendering is reserved to the organizers, in the
training mode as in the contest mode.
"""

import argparse
import pathlib
import sys
from typing import Any

from gymnasium import spaces
import numpy as np
import pettingzoo

# Importing the client library registers the remote environments in the
# PettingZoo registry
import pzclient

# The remote world of the contest, served by "pettingzoo-server"
ENV_ID = "csc53439ep/csc53439ep-remote-v1"


def policy(observation: np.ndarray, action_space: spaces.Space) -> Any:
    """
    Choose the action of one agent.

    **This is the function to replace by your own agent.**  The observation is
    a vector of floats and the action a binary vector: see the observation and
    action spaces of the agent.

    Parameters
    ----------
    observation : numpy.ndarray
        The observation of the agent, a sample of its observation space.
    action_space : gymnasium.spaces.Space
        The action space of the agent, the actions must belong to it.

    Returns
    -------
    Any
        The action of the agent, a sample of `action_space`.
    """
    return action_space.sample()


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    parser.add_argument("--api-url", "-u", default=None,
                        help="base URL of the API of the contest server "
                             "(default: the PETTINGZOO_API_URL variable, else "
                             f"{pzclient.DEFAULT_API_URL})")
    parser.add_argument("--token", "-t", default=None,
                        help="the token given to you by the organizers "
                             "(default: the PETTINGZOO_TOKEN variable)")
    parser.add_argument("--steps", "-n", type=int, default=200,
                        help="number of steps to play; the episode is eternal, "
                             "so the loop has to be bounded (default: 200)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="seed of the random policy, to make it reproducible")
    parser.add_argument("--frames-dir", "-f", type=pathlib.Path, default=None,
                        help="directory where the rendered frames of the world "
                             "are saved as PNG images (with an organizer "
                             "token only)")
    parser.add_argument("--render-mode", "-r", default="rgb_array",
                        choices=["human", "rgb_array", "none"],
                        help="'human' to watch the world in a window refreshed "
                             "at every step (needs pygame), 'rgb_array' to "
                             "render only the frames of --frames-dir, 'none' "
                             "to render nothing (default: rgb_array); "
                             "'human' needs an organizer token")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="only print the summary of the episode")

    args = parser.parse_args()

    if args.render_mode == "none" and args.frames_dir is not None:
        parser.error("--frames-dir needs a render mode, not 'none'")

    return args


def main() -> int:
    """
    Play the world with a random policy and report what happened.

    Returns
    -------
    int
        The exit status of the script.
    """
    args = parse_args()

    # Connect to the remote environment.  This is the only line that differs
    # from a local PettingZoo environment: the URL of the server and the token
    # of the participant have to be given.
    try:
        env = pettingzoo.make(
            "parallel",
            ENV_ID,
            api_url=args.api_url,
            token=args.token,
            render_mode=None if args.render_mode == "none" else args.render_mode,
        )
    except pzclient.RemoteEnvError as error:
        print(f"Cannot connect to the environment: {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        # The "human" render mode needs pygame, an optional dependency
        print(error, file=sys.stderr)
        return 1

    print(f"Server:      {env.api_url}")
    print(f"Your agents: {', '.join(env.possible_agents)}")
    print(f"Observation: {env.observation_space(env.possible_agents[0])}")
    print(f"Action:      {env.action_space(env.possible_agents[0])}")
    print()

    if args.seed is not None:
        # Seed the action spaces to make the random policy reproducible
        for offset, agent in enumerate(sorted(env.possible_agents)):
            env.action_space(agent).seed(args.seed + offset)

    if args.frames_dir is not None:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    cumulative_rewards: dict[str, float] = {}
    step = 0

    try:
        # Connect the agent to the shared world and get its first observation
        observations, _ = env.reset()

        # The episode is eternal: the loop is bounded by "--steps" rather than
        # by the usual "while env.agents:"
        while env.agents and step < args.steps:
            actions = {
                agent: policy(observations[agent], env.action_space(agent))
                for agent in env.agents
            }

            observations, rewards, terminations, truncations, _ = env.step(actions)
            step += 1

            for agent, reward in rewards.items():
                cumulative_rewards[agent] = cumulative_rewards.get(agent, 0.0) + reward

            if any(terminations.values()) or any(truncations.values()):
                # The episode of the contest world is eternal, so this should
                # not happen; the parallel API allows it, though, and the
                # agents that are done have left "env.agents"
                print("Some of your agents are done: they left the episode.")

            if args.frames_dir is not None:
                try:
                    frame = env.render_png()
                except pzclient.PermissionDeniedError:
                    # Rendering is reserved to the organizers
                    print("Your token does not allow rendering the world: "
                          "no frame will be saved.", file=sys.stderr)
                    args.frames_dir = None
                else:
                    if frame is not None:
                        (args.frames_dir / f"frame_{step:05d}.png").write_bytes(frame)

            if not args.quiet:
                print(f"step {step:>5} | "
                      f"reward: {sum(rewards.values()):>+8.3f} | "
                      f"total: {sum(cumulative_rewards.values()):>+9.3f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except pzclient.RemoteEnvError as error:
        print(f"The episode was interrupted: {error}", file=sys.stderr)
        return 1
    finally:
        # Always close the environment: this is how your agent leaves the
        # shared world, instead of staying connected until the server drops it
        env.close()

    print()
    print(f"Played {step} steps.")

    for agent in sorted(cumulative_rewards):
        print(f"  {agent}: total reward {cumulative_rewards[agent]:+.3f}")

    if args.frames_dir is not None:
        print(f"Frames saved in {args.frames_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
