# Playing the contest with `pzclient`

This tutorial explains how to connect your agent to the world of the contest, and how that
world behaves, so that your agent plays it correctly. Some familiarity with the [PettingZoo parallel
API](https://pettingzoo.farama.org/api/parallel/) helps.

The world (i.e. the environment) runs on the server of the organizers; your agent runs on your machine. `pzclient` hides
the network behind the standard PettingZoo API, so the remote world is used like a local
environment. There are a few differences, though, and they are what this tutorial is about.

## 1. Install and connect

The organizers give you your **token**. It identifies you on the server and names your agent in
the world: keep it for yourself. The same token works on the two servers of the organizers, one
per mode (see [section 3](#3-training-mode-and-contest-mode)):

| Server | URL (`api_url`) |
|---|---|
| Training | `https://csc53439ep.jdhp.org/training/api` |
| Contest | `https://csc53439ep.jdhp.org/api` |

The library itself is installed from the GitHub release of `pzclient` (Python 3.12 or later),
preferably in a virtual environment of its own:

```sh
python3 -m venv env          # create the virtual environment, once
source env/bin/activate      # activate it, in every new shell (Windows: .venv\Scripts\activate)

pip install https://github.com/jeremiedecock/pettingzoo-client/releases/download/v0.4.0/pzclient-0.4.0-py3-none-any.whl
```

Once it is installed, the remote world is built with the standard `pettingzoo.make`, from the URL of
a server and your token. You can then read the agents you control and their observation and action
spaces:

```python
import pettingzoo
import pzclient  # importing the library registers the remote environments

env = pettingzoo.make(
    "parallel",
    "csc53439ep/csc53439ep-remote-v1",
    api_url="https://csc53439ep.jdhp.org/training/api",  # the contest: https://csc53439ep.jdhp.org/api
    token="<your token>",
)

print(env.possible_agents)        # the agents you control
print(env.observation_space(env.possible_agents[0]))
print(env.action_space(env.possible_agents[0]))
```

Instead of passing them each time, you can export the URL and the token once; they are the defaults
of `api_url` and `token`:

```sh
export PETTINGZOO_API_URL=https://csc53439ep.jdhp.org/training/api  # the contest: https://csc53439ep.jdhp.org/api
export PETTINGZOO_TOKEN=<your token>
```

Building the environment already talks to the server, so it fails at once if something is wrong:
`pzclient.AuthenticationError` if your token is refused, and `pzclient.ServerUnreachableError` if the
server cannot be reached.

## 2. The world at a glance

The world is a **black box**. This tutorial describes the data your agent exchanges with it, not
what that data means: your agent only knows the world through its observations and its rewards,
and so do you.

**Observation**: a `Box(0, 1, (11,), float32)`, i.e. a vector of 11 floats between 0 and 1.

The first observation, the one `reset()` returns, is all zeros: the observation is only filled once
a step has been played.

**Action**: a `MultiBinary(2)`, i.e. a vector of 2 integers, each 0 or 1: `[0, 0]`, `[0, 1]`,
`[1, 0]` or `[1, 1]`.

**Reward**: a float, `rewards[agent]`, returned by each step.

**No agent is ever terminated**: `terminations[agent]` is always `False`.

**The episode is eternal**: no agent ever leaves `env.agents`, so `while env.agents:` never ends.
Always bound your loops.

## 3. Training mode and contest mode

The organizers run two servers, one per mode, and your token works on both. Your code is the same
for both, only its loop differs: to switch from one to the other, change `api_url`.

| | Training mode | Contest mode |
|---|---|---|
| `api_url` | `https://csc53439ep.jdhp.org/training/api` | `https://csc53439ep.jdhp.org/api` |
| Your world | a world of your own, nobody else in it | one world shared by all the participants |
| Your agents | all the agents of the world (`agent_0`, `agent_1`, …) | a single agent, named after you |
| `reset()` | resets your world (`seed` works) | connects a brand new agent of yours to the running world |
| `step()` | plays the step at once | waits for the world to play the step, on its own clock |
| Time | stands still until you call `step()` | goes on without you |
| `close()` | discards your world | disconnects your agent; the world goes on |
| `state()` | allowed | forbidden (`PermissionDeniedError`) |
| `render()` | forbidden (`PermissionDeniedError`) | forbidden (`PermissionDeniedError`) |

You can tell them apart from `env.possible_agents`: several `agent_<n>` in the training mode, your
own name alone in the contest mode.

### Training mode

The world is yours, and it waits for you. It only advances when you call `step()`, so you may take
as long as you want to choose your actions: nothing happens in the meantime. Use it to train, with
all its agents if you like.

- `step()` needs **exactly one action per agent** of `env.agents`: a missing or invalid action
  raises `pzclient.InvalidActionError`.
- `close()` discards your world; the next `reset()` starts a fresh one.
- Your world lives on the server as long as the server runs: a restart of the server loses it.
- Your world belongs to your token, not to your program: the server cannot tell apart two programs
  using the same token. **Do not run two programs with the same token** (two scripts, two notebooks, or a script
  and a notebook) on the training server: they drive the same world, and interfere with each other.
  Their steps interleave, so each program sees the world jump ahead between two of its calls, and
  receives observations and rewards that the actions of the other program shaped too; a `reset()`
  of one program resets the world of the other; and the `close()` of the first program to finish
  discards the world of the other, whose next `step()` raises `pzclient.AgentNotConnectedError`.

**Rendering is reserved to the organizers**, in the training mode as in the contest mode:
`render()`, and `render_mode="human"`, raise `pzclient.PermissionDeniedError` with your token.
The world is **partially observable** (a POMDP): each agent only perceives a small part of it,
through its observation. A picture of the world would give you far more than what your agents see,
and the challenge is precisely to know no more about the world than they do: your agents only know
it through their observations, and so do you.

#### A simple training loop

This complete program resets a world of your own, drives all its agents with random actions for
500 steps, then discards the world.

```python
import pettingzoo
import pzclient

env = pettingzoo.make(
    "parallel",
    "csc53439ep/csc53439ep-remote-v1",
    api_url="https://csc53439ep.jdhp.org/training/api",
    token="<your token>",
)

observations, infos = env.reset(seed=42)  # a fresh world of your own

for _ in range(500):  # the episode is eternal: bound the loop
    # One action per agent of env.agents: replace the random actions by your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)

env.close()  # discard your world
```

### Contest mode

In the contest mode, all the participants play **the same world at the same time**, and you
control **one agent**, named after you. The other participants join and leave at any moment, so the
number of agents in the world changes while you play.

- `reset()` connects a brand new agent of yours to the world. It does *not* reset the world, and
  its `seed` and `options` arguments are ignored. If you already had an agent, it is replaced: you
  never control two agents at once in the contest mode.
- `close()` disconnects your agent. The world goes on for the others, and you may join again later.
- `render()` and `state()` are reserved to the organizers: they would show you more of the world
  than your agent observes. They raise `pzclient.PermissionDeniedError`; your agent only knows the
  world through its observations.

#### The world has its own clock

This is the most important difference with a local environment. Time is cut into **windows** of `T`
seconds, and each window ends with one step of the world. `T` is 5 seconds for now, but it may
change depending on the lag observed between the clients and the server; the organizers will let
you know if it does.

1. A window opens as soon as the previous step has been played, at the same moment for everybody.
2. If **every** connected participant sends its action before the end of the window, the step is
   played at once, without waiting for the end of the window.
3. Otherwise, the step is played at the end of the window. The agents whose action has not arrived
   play a default action. Even if nobody answers, the world keeps going.

Your `step()` call returns once the world has played the step, with what your agent observes after it.
It therefore blocks for at most `T` seconds, and returns quickly when everybody is fast (alone in
the world, you play as fast as you answer).

**Your time budget.** Between the moment `step()` returns and the end of the next window, you have
`T` seconds minus the network round trip, for your agent to choose its next action. The clock starts
when the previous step is played, not when your action arrives. If your agent is too slow, it plays
the default action during that step, and the world goes on without you.

#### Late actions are ignored

`pzclient` tells the server which step each of your actions is meant for. If an action arrives after
its step has been played, the server **ignores** it: it never applies an old action to a later step.
`step()` then returns **at once**, with the latest observation of your agent, so that it can
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

#### When your agent is disconnected

Your agent is removed from the world when:

- it misses **12 steps in a row** (about one minute with `T = 5 s`): your program crashed, stopped
  without calling `close()`, or is always slower than `T`. An ignored action counts as a miss;
- the server restarts (the organizers will announce it). While it restarts, your calls fail with
  `pzclient.ServerUnreachableError`, or with a `pzclient.RemoteEnvError` whose `status_code` is
  502, 503 or 504.

Your next `step()` then raises `pzclient.AgentNotConnectedError`: call `reset()` to connect a new
agent.

#### A simple contest loop

This complete program joins the shared world, plays 500 steps with random actions, then leaves. It
does not handle disconnections: see the next section for that.

```python
import pettingzoo
import pzclient

env = pettingzoo.make(
    "parallel",
    "csc53439ep/csc53439ep-remote-v1",
    api_url="https://csc53439ep.jdhp.org/api",
    token="<your token>",
)
agent = env.possible_agents[0]  # in the shared world, your agent is named after you

observations, infos = env.reset()  # join the shared world

for _ in range(500):  # the episode is eternal: bound the loop
    actions = {agent: env.action_space(agent).sample()}  # replace by your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)

env.close()  # leave the shared world
```

#### A more robust contest loop

```python
import time

import pettingzoo
import pzclient


def policy(observation):
    """Your agent: an observation in, an action out (here, always the same one)."""
    return [1, 1]


def is_transient(error):
    """Whether an error is worth waiting for: the network is down, or the server is restarting."""
    return (
        isinstance(error, pzclient.ServerUnreachableError)
        or error.status_code in (502, 503, 504)
    )


def join(env):
    """Connect a new agent to the world, waiting for the server if needed."""
    while True:
        try:
            return env.reset()
        except pzclient.RemoteEnvError as error:
            if not is_transient(error):
                raise
            time.sleep(5)


env = pettingzoo.make("parallel", "csc53439ep/csc53439ep-remote-v1")  # PETTINGZOO_API_URL, PETTINGZOO_TOKEN
agent = env.possible_agents[0]
observations, infos = join(env)

for _ in range(10_000):  # the episode is eternal: bound the loop
    action = policy(observations[agent])

    try:
        observations, rewards, terminations, truncations, infos = env.step({agent: action})
    except pzclient.AgentNotConnectedError:
        # Your agent left the world (see above): connect a new one
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

env.close()
```

## 4. Schedule of the contest

Both servers, training and contest, are already open, and your token works on both.

- **Train your agents in the training mode.** It is made for that: the world is yours, it waits for
  you, and you drive all its agents (see [Training mode](#training-mode)).
- **The contest mode is open for testing during the first four weeks of the course.** Use it to
  check that your agent plays the shared world correctly (its clock, the other participants, the
  disconnections). During these four weeks, the organizers ignore the results of your agents in
  that mode.
- **The contest officially takes place during the fifth week of the course**, from **Monday 19
  October 2026, 00:00**, to **Wednesday 21 October 2026, 23:59** (Paris time).
- **The organizers may add agents of their own to the shared world** at any time, whenever they
  see fit.

## 5. Tips

- **Bound your loops**: the episode never ends by itself.
- **Keep your agent fast** in the contest mode. Measure how long your policy takes
  (`time.monotonic()` before and after the call) and keep it well below `T`. Watch
  `action_applied`: if it is often `False`, your agent is too slow; run it on a faster machine (e.g.
  Google Colab) or simplify it.
- **One program per token and per server.** Two programs sharing a token on the same server fight
  over the same agent (contest) or the same world (training); one on each server is fine.
- **Do not lower the HTTP timeout** of the environment (`timeout`, 60 s by default) below `2T + 1`
  seconds: in the contest mode `step()` legitimately waits for up to `T` seconds, and for up to
  `2T + 1` seconds before the server reports that the world could not play the step (503).
- **Call `close()`** when you stop playing (or use `with pettingzoo.make(...) as env:`), so that
  your agent leaves the world at once instead of playing the default action for a minute.

## 6. Errors

Everything `pzclient` raises derives from `pzclient.RemoteEnvError`.

| Exception | Cause | What to do |
|---|---|---|
| `AuthenticationError` | your token is missing, wrong or revoked | check your token |
| `PermissionDeniedError` | your token is valid, but does not grant the request: `render()`, or `state()` in the contest mode | do not call them |
| `AgentNotConnectedError` | no agent of yours is in the world: not joined yet, disconnected, the server restarted, or another program using your token called `close()` | call `reset()` (and stop the other program) |
| `InvalidActionError` | the actions do not match your agents, are outside their action space, or cannot be encoded as JSON (a NaN in a plain list) | fix the actions |
| `ServerUnreachableError` | the network is down, or the server is restarting | wait and retry |
| `RemoteEnvError` | anything else; with a `status_code` of 502, 503 or 504, the server is restarting or could not play the step in time | wait and retry; tell the organizers if it lasts |
