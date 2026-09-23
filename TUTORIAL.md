# Playing the contest with `pzclient`

This tutorial explains how to connect your agent to the `alife` world of the contest, and how that
world behaves, so that your agent plays it correctly. It assumes that you know Python and the basics
of reinforcement learning; some familiarity with the [PettingZoo parallel
API](https://pettingzoo.farama.org/api/parallel/) helps.

The world runs on the server of the organizers; your agent runs on your machine. `pzclient` hides
the network behind the standard PettingZoo API, so the remote world is used like a local
environment. There are a few differences, though, and they are what this tutorial is about.

## 1. Install and connect

The organizers give you three things: the wheel of `pzclient`, the URL of the server, and your
**token**. The token identifies you on the server and names your agent in the world: keep it for
yourself.

```sh
pip install pzclient-0.2.0-py3-none-any.whl   # Python 3.12 or later
```

```python
import pettingzoo
import pzclient  # importing the library registers the remote environments

env = pettingzoo.make(
    "parallel",
    "alife/alife-remote-v1",
    api_url="<the URL given by the organizers>",  # e.g. "https://.../api"
    token="<your token>",
)

print(env.possible_agents)        # the agents you control
print(env.observation_space(env.possible_agents[0]))
print(env.action_space(env.possible_agents[0]))
```

Instead of passing them each time, you can export the URL and the token once; they are the defaults
of `api_url` and `token`:

```sh
export PETTINGZOO_API_URL=<the URL given by the organizers>
export PETTINGZOO_TOKEN=<your token>
```

Building the environment already talks to the server, so it fails at once if something is wrong:
`pzclient.AuthenticationError` if your token is refused, `pzclient.ServerUnreachableError` if the
server cannot be reached, and `pzclient.RemoteEnvError` if the server does not serve the `alife`
world.

[`examples/random_agents.py`](examples/random_agents.py) is a complete, ready-to-run agent, with a
`policy` function to replace by yours:

```sh
python examples/random_agents.py --steps 500   # uses PETTINGZOO_API_URL and PETTINGZOO_TOKEN
```

## 2. The world at a glance

Your agent is a **bug** living on an island. The figures below are those of `alife` 0.1.0.

**Observation**: a `Box(0, 1, (11,), float32)`, the sensors of the bug.

| Index | Sensor |
|---|---|
| 0 | collision: 1 when the body of the bug touches something (a rock, a plant, another bug), else 0 |
| 1 | proximity: how close the bug is to a wall or to another sprite (0 = nothing around, 1 = touching) |
| 2–4 | antenna 1, on the side `[1, 0]` turns to: the colour at its tip, as RGB — black: nothing, blue: rock or wall, green: plant, red: bug |
| 5–7 | antenna 2, on the side `[0, 1]` turns to: the same |
| 8 | energy: the health of the bug, between 0 and 1 |
| 9 | compass: how well the bug points at its target flag, the cosine of the angle between them (1: straight at it, 0: 90° off or more) |
| 10 | speed, normalized between 0 and 1 |

The first observation, the one `reset()` returns, is all zeros: a bug only senses the world once a
step has been played.

**Action**: a `MultiBinary(2)`, `[R, L]`.

| Action | Effect |
|---|---|
| `[1, 1]` | full speed ahead |
| `[1, 0]` | turn right, at half speed |
| `[0, 1]` | turn left, at half speed |
| `[0, 0]` | no thrust, no turn: the bug brakes, and slides until it stops |

The bug accelerates and brakes progressively: an action changes its speed, it does not set it.

**Reward**: `+5` when the spear of the bug (its tip, at the front) touches its **target flag**,
which then moves on to the next flag; `-5` when the bug dies. The flags are invisible to the
antennae: the compass is the only way to find them.

**Death and rebirth.** A bug dies when it enters a wall or the water, or when its energy runs out.
Living costs a little energy at each step, crashing into something costs energy (the faster the
bug, the more), and being speared or bumped by another bug costs energy too. Spearing a plant takes
a bite of it, which restores energy. A dead bug is reborn at once at a nest, with half its energy,
and keeps playing: a death is reported by `infos[agent]["died"]` and by the `-5` reward, **never**
by a termination.

**The episode is eternal**: no agent ever leaves `env.agents`, so `while env.agents:` never ends.
Always bound your loops.

## 3. Training mode and contest mode

The server runs in one of two modes. The organizers tell you which server is which; your code is the
same for both, only its loop differs.

| | Training mode | Contest mode |
|---|---|---|
| Your world | a world of your own, nobody else in it | one world shared by all the participants |
| Your agents | all the bugs of the world (`agent_0`, `agent_1`, …) | a single bug, named after you |
| `reset()` | resets your world (`seed` works) | connects a brand new bug of yours to the running world |
| `step()` | plays the step at once | waits for the world to play the step, on its own clock |
| Time | stands still until you call `step()` | goes on without you |
| `close()` | discards your world | disconnects your bug; the world goes on |
| `state()` | allowed | forbidden (`AuthenticationError`) |

You can tell them apart from `env.possible_agents`: several `agent_<n>` in the training mode, your
own name alone in the contest mode.

### Training mode

The world is yours, and it waits for you. It only advances when you call `step()`, so you may take
as long as you want to choose your actions: nothing happens in the meantime. Use it to train, with
all its bugs if you like.

```python
observations, infos = env.reset(seed=42)

for _ in range(10_000):  # the episode is eternal: bound the loop
    actions = {agent: policy(observations[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if not env.agents:  # only if the organizers set a maximum episode length
        observations, infos = env.reset()

env.close()
```

- `step()` needs **exactly one action per agent** of `env.agents`: a missing or invalid action
  raises `pzclient.InvalidActionError`.
- `close()` discards your world; the next `reset()` starts a fresh one.
- Your world lives on the server as long as the server runs: a restart of the server loses it.
- Do not run two programs with the same token: they would drive the same world, and interfere with
  each other.

### Contest mode

In the contest mode, all the participants play **the same world at the same time**, and you
control **one bug**, named after you. The other participants join and leave at any moment, so the
number of bugs around you changes while you play.

- `reset()` connects a brand new bug of yours to the world, at a nest. It does *not* reset the
  world, and its `seed` and `options` arguments are ignored. If you already had a bug, it is
  replaced: you never control two bugs at once.
- `close()` disconnects your bug. The world goes on for the others, and you may join again later.
- `render()` returns a picture of the whole world, the same for everybody.

#### The world has its own clock

This is the most important difference with a local environment. Time is cut into **windows** of `T`
seconds (the organizers announce `T`; 5 seconds by default), and each window ends with one step of
the world:

1. A window opens as soon as the previous step has been played, at the same moment for everybody.
2. If **every** connected participant sends its action before the end of the window, the step is
   played at once, without waiting for the end of the window.
3. Otherwise, the step is played at the end of the window. The bugs whose action has not arrived
   play `[0, 0]` during that step: they brake. Even if nobody answers, the world keeps going.

Your `step()` call returns once the world has played the step, with what your bug senses after it.
It therefore blocks for at most `T` seconds, and returns quickly when everybody is fast (alone in
the world, you play as fast as you answer).

**Your time budget.** Between the moment `step()` returns and the end of the next window, you have
`T` seconds minus the network round trip, for your agent to choose its next action. The clock starts
when the previous step is played, not when your action arrives. If your agent is too slow, your bug
brakes during that step, and the world goes on without you.

#### Late actions are ignored

`pzclient` tells the server which step each of your actions is meant for. If an action arrives after
its step has been played, the server **ignores** it: it never applies an old action to a later step.
`step()` then returns **at once**, with the latest observation of your bug, so that your agent can
catch up. The same happens if you send a second action during the same window: only the first one is
played.

The server tells you what became of each action in `infos[agent]["shared_world"]`:

| Key | Meaning |
|---|---|
| `action_applied` | `True` if your action was played in the step, `False` if it was ignored |
| `message` | why it was ignored, or `None` |
| `step` | the number of the next step of the world (useful to see that you missed some) |

If you miss several steps in a row, the observation and the reward you get are those of the last
step only: the observations and the rewards of the steps before it are lost for you.

#### When your bug is disconnected

Your bug is removed from the world when:

- it misses **12 steps in a row** (about one minute with `T = 5 s`): your program crashed, stopped
  without calling `close()`, or is always slower than `T`. An ignored action counts as a miss;
- the server restarts (the organizers will announce it). While it restarts, your calls fail with
  `pzclient.ServerUnreachableError`, or with a `pzclient.RemoteEnvError` whose `status_code` is
  502, 503 or 504.

Your next `step()` then raises `pzclient.AgentNotConnectedError`: call `reset()` to connect a new
bug.

#### A robust contest loop

```python
import time

import pettingzoo
import pzclient


def policy(observation):
    """Your agent: an observation in, an action out (here, full speed ahead)."""
    return [1, 1]


def is_transient(error):
    """Whether an error is worth waiting for: the network is down, or the server is restarting."""
    return (
        isinstance(error, pzclient.ServerUnreachableError)
        or error.status_code in (502, 503, 504)
    )


def join(env):
    """Connect a new bug to the world, waiting for the server if needed."""
    while True:
        try:
            return env.reset()
        except pzclient.RemoteEnvError as error:
            if not is_transient(error):
                raise
            time.sleep(5)


env = pettingzoo.make("parallel", "alife/alife-remote-v1")  # PETTINGZOO_API_URL, PETTINGZOO_TOKEN
agent = env.possible_agents[0]
observations, infos = join(env)

for _ in range(10_000):  # the episode is eternal: bound the loop
    action = policy(observations[agent])

    try:
        observations, rewards, terminations, truncations, infos = env.step({agent: action})
    except pzclient.AgentNotConnectedError:
        # Your bug left the world (see above): connect a new one
        observations, infos = join(env)
        continue
    except pzclient.RemoteEnvError as error:
        if not is_transient(error):
            raise  # a wrong token or an invalid action: fix your code
        time.sleep(5)
        continue

    shared_world = infos[agent]["shared_world"]
    if not shared_world["action_applied"]:
        print(f"step {shared_world['step']}: {shared_world['message']}")

    if infos[agent]["died"]:
        print("My bug died, and was reborn at a nest")

env.close()
```

## 4. Tips

- **Bound your loops**: the episode never ends by itself.
- **Keep your agent fast** in the contest mode. Measure how long your policy takes
  (`time.monotonic()` before and after the call) and keep it well below `T`. Watch
  `action_applied`: if it is often `False`, your agent is too slow; run it on a faster machine (e.g.
  Google Colab) or simplify it.
- **One program per token.** Two programs sharing a token fight over the same bug (contest) or the
  same world (training).
- **Render sparingly** in the contest mode: the server draws the picture of the whole world, and the
  world waits while it does. Call `render()` from time to time, not at every step.
- **Do not lower the HTTP timeout** of the environment (`timeout`, 60 s by default) below `T`: in
  the contest mode `step()` legitimately waits for up to `T` seconds.
- **Call `close()`** when you stop playing (or use `with pettingzoo.make(...) as env:`), so that
  your bug leaves the world at once instead of braking in the way of the others for a minute.

## 5. Errors

Everything `pzclient` raises derives from `pzclient.RemoteEnvError`.

| Exception | Cause | What to do |
|---|---|---|
| `AuthenticationError` | your token is missing, wrong or revoked; or `state()` in the contest mode | check your token |
| `AgentNotConnectedError` | no bug of yours is in the world: not joined yet, disconnected, or the server restarted | call `reset()` |
| `InvalidActionError` | the actions do not match your agents, or are outside their action space | fix the actions |
| `ServerUnreachableError` | the network is down, or the server is restarting | wait and retry |
| `RemoteEnvError` | anything else; with a `status_code` of 502, 503 or 504, the server is restarting or could not play the step in time | wait and retry; tell the organizers if it lasts |
