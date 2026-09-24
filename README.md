# PettingZoo client

The client library of [`pettingzoo-server`](https://github.com/jeremiedecock/pettingzoo-server):
it makes a PettingZoo environment served over a REST API usable like a local one.

This is the library the participants of the MARL contest install to connect their agent to the
shared [`alife`](https://github.com/jeremiedecock/alife) world hosted by the organizers. The
environment runs on the server, your agent runs on your machine, and everything in between is
hidden behind the official [PettingZoo parallel
API](https://pettingzoo.farama.org/api/parallel/).

New to the contest? Read the [tutorial](TUTORIAL.md) first: it explains how the world of the contest
behaves, and how to write an agent that plays it correctly.

## Installation

```sh
pip install https://github.com/jeremiedecock/pettingzoo-client/releases/download/v0.3.1/pzclient-0.3.1-py3-none-any.whl
```

The wheel is attached to the [GitHub releases](https://github.com/jeremiedecock/pettingzoo-client/releases)
of this repository; the organizers give you your token. Python 3.12 or later is required; `gymnasium`, `numpy`, `pettingzoo`, `pillow` and `requests` are installed as
dependencies. `pygame` is optional: install it (`pip install pygame`) to watch the world in a window
with `render_mode="human"` (organizers only, see below).

## The servers

The organizers run two servers, one per mode; your token works on both:

| Server | URL (`api_url`) | What you get |
|---|---|---|
| Training | `https://csc53439ep.jdhp.org/training/api` | a world of your own, where you drive all the bugs (4 of them, `agent_0` to `agent_3`: the server sets their number, and you cannot change it), and which only advances when you call `step()` |
| Contest | `https://csc53439ep.jdhp.org/api` | one bug, named after you, in the world shared by all the participants |

The [tutorial](TUTORIAL.md#3-training-mode-and-contest-mode) explains the differences between the
two modes.

## Quick start

```python
import pettingzoo
import pzclient  # importing the library registers the remote environments

env = pettingzoo.make(
    "parallel",
    "alife/alife-remote-v1",
    api_url="https://csc53439ep.jdhp.org/training/api",  # the contest: https://csc53439ep.jdhp.org/api
    token="<your token>",
)

observations, infos = env.reset()

# The episode is eternal: bound your loop (see "What is special" below)
for _ in range(1000):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

That is the usual PettingZoo parallel loop: only the two arguments of `make` are specific to the
remote environment. `pzclient.RemoteParallelEnv(api_url=..., token=...)` builds the same
environment without the registry.

[`examples/random_agents.py`](examples/random_agents.py) is the same thing, ready to run, with a
`policy` function to replace by your agent:

```sh
python examples/random_agents.py --token <your token> \
                                 --api-url https://csc53439ep.jdhp.org/training/api \
                                 --steps 500
```

## Your token

Every participant receives a token from the organizers; it identifies you on the server and names
your agent in the shared world. Pass it as the `token` argument or export it once:

```sh
export PETTINGZOO_API_URL=https://csc53439ep.jdhp.org/training/api  # the contest: https://csc53439ep.jdhp.org/api
export PETTINGZOO_TOKEN=<your token>
```

These two environment variables are the defaults of the `api_url` and `token` arguments.

The server knows you by your token alone, not by your program: two programs running with the same
token on the same server (two scripts, or a script and a notebook) share the same world in the
training mode and the same bug in the contest mode, and disrupt each other. Run one program per
token and per server; the [tutorial](TUTORIAL.md#training-mode) details what goes wrong otherwise.

## The registered environments

| Id | Environment |
|---|---|
| `alife/alife-remote-v1` | The `alife` world of the contest; the constructor fails if the server serves anything else. |
| `remote/parallel-v1` | Whatever parallel environment the server serves (useful to try your agent against an official PettingZoo environment). |

## The API

`RemoteParallelEnv` is a `pettingzoo.ParallelEnv` and implements the whole parallel API:
`possible_agents`, `agents`, `num_agents`, `max_num_agents`, `observation_space(agent)`,
`action_space(agent)`, `reset(seed, options)`, `step(actions)`, `render()`, `state()` and
`close()`. It passes `pettingzoo.test.parallel_api_test`, and it is also a context manager, so
`with pettingzoo.make(...) as env:` closes the environment for you.

The server always draws the world as a PNG image; the `render_mode` argument of `make` chooses what
the client does with it:

| `render_mode` | Rendering |
|---|---|
| `"rgb_array"` (default) | `render()` returns the frame as a `(H, W, 3)` uint8 numpy array, e.g. to show it in a Jupyter notebook. |
| `"human"` | The frame is shown in a pygame window, refreshed after every `reset()` and `step()`; `render()` returns `None`. The server sends JPEG images, much lighter than the PNG ones of `"rgb_array"`. The window can be resized, and closing it stops the rendering (`render_mode` becomes `None`) while the agents keep playing. Requires `pygame`. |
| `None` | Nothing is rendered. |

Rendering is reserved to the organizers, in the training mode as in the contest mode: for a
participant, `render()` (and thus `reset()` and `step()` with `render_mode="human"`) raises
`PermissionDeniedError`.

Two additions are specific to the remote environment:

| Method | Role |
|---|---|
| `render_png()` | The frame as the PNG bytes the server sends, handy to save a picture of the world without re-encoding it. |
| `remote_agents()` | Reads `agents` from the server instead of using the local copy, handy after a network error. |

Everything the library raises derives from `pzclient.RemoteEnvError`:
`AuthenticationError` (your token is missing or refused), `PermissionDeniedError` (your token does
not grant the request, e.g. `state()` in the contest mode), `AgentNotConnectedError` (no agent of
yours is acting: call `reset()`), `InvalidActionError` (the actions do not match the acting agents
or their spaces) and `ServerUnreachableError` (the server could not be reached).

## What is special about the contest environment

The `alife` world served during the contest departs from the usual PettingZoo assumptions on three
points, and your agent has to be written accordingly:

- **The episode is eternal.** A bug that dies is reborn at a nest and keeps playing, so no agent
  ever leaves `agents` and `while env.agents:` never ends. Bound your loop. A death is reported by
  `infos[agent]["died"]` and by the death penalty in the reward, not by a termination.
- **The world is shared, and you control a single agent.** All the participants play the same
  world at the same time; `possible_agents` holds your agent alone, named after you. `reset()`
  connects a brand new bug of yours to the running world (it does *not* reset the world, and its
  `seed` and `options` arguments are ignored), and `close()` disconnects it. The participants join
  and leave at any moment, so the number of bugs around you changes while you play.
- **The world has its own clock.** Time is cut into windows of a few seconds (the step timeout of
  the server): a step is played at the end of each window, or as soon as every connected
  participant has sent its action, and `step()` blocks until then. An agent that has not answered
  in time simply does nothing during that step, and the world goes on without it. An action that
  arrives after its step has been played is ignored (`step()` then returns at once, with the
  latest observation of your agent), and an agent that stops answering is eventually disconnected
  from the world. `infos[agent]["shared_world"]` tells what became of your action:
  `action_applied` (whether it was played), `message` (why it was not) and `step` (the number of
  the next step of the world).

## Development

```sh
pip install -r requirements-dev.txt   # installs the library in editable mode
pytest                                # the tests need no server
ruff check pzclient/                  # lint
python -m build --wheel               # build the wheel given to the participants
```

The tests replace the HTTP session of the environment by a stand-in of the server, so they run
offline; `pettingzoo-server` is the place where the two halves are tested together.

### Publishing a new version

The wheel is not on PyPI: the [`Publish wheel`](.github/workflows/publish-wheel.yml) GitHub
Actions workflow attaches it to a GitHub release every time a `v*` tag is pushed. To publish
version `X.Y.Z`:

1. Run `just bump X.Y.Z`: it sets `version = "X.Y.Z"` in [`pyproject.toml`](pyproject.toml) and
   updates the `pip install` URLs of this README, of the [tutorial](TUTORIAL.md) and of the
   [examples](examples/).
2. Commit and push, then tag that commit and push the tag:

   ```sh
   git commit -am "Bump the version to X.Y.Z."
   git push
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The workflow builds the wheel and the source distribution (`python -m build`), creates the
`vX.Y.Z` release with generated notes, and attaches both files to it. The wheel can then be
installed from
`https://github.com/jeremiedecock/pettingzoo-client/releases/download/vX.Y.Z/pzclient-X.Y.Z-py3-none-any.whl`.
The tag must match `version`, and nothing checks it: the name of the release comes from the tag,
the name of the wheel from `pyproject.toml`.

Re-running the workflow of an existing tag (*Re-run jobs* on its run, in the *Actions* tab)
replaces the files of the release. Started by hand from a branch (*Run workflow*), it only builds
the distributions, and leaves them as an artifact of the run.
