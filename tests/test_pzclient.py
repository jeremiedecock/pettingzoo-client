"""
Tests of the client library.

They need no server: the HTTP session of `pzclient.RemoteParallelEnv` is
replaced by `FakeSession`, a stand-in of `pettingzoo-server` that answers the
endpoints of the parallel API with the payloads the server would send.  What is
checked here is the client half of the contract: the requests it sends, the
codec of the numpy objects and of the spaces, the parallel API it exposes and
the errors it raises.

Run them with ``pytest`` from the root directory of the project.
"""

import io
import json

from gymnasium import spaces
import numpy as np
import pettingzoo
from PIL import Image
import pytest
import requests

import pzclient
from pzclient import serialization

# The participant and its single agent, as the contest server answers them
TOKEN = "token_abc123"
AGENT = "alice"

# The environment served by `FakeSession`
ENV_ID = "alife/alife-v1"
OBSERVATION_SPACE = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
ACTION_SPACE = spaces.MultiBinary(2)


###############################################################################
# THE STAND-IN OF THE SERVER ##################################################
###############################################################################


class FakeResponse:
    """
    The answer of `FakeSession`, with the interface `requests` gives.

    Parameters
    ----------
    payload : dict, optional
        The JSON body of the response.
    content : bytes, optional
        The raw body of the response, for the binary answers.
    media_type : str
        The media type of `content`.
    status_code : int
        The HTTP status of the response.
    """

    def __init__(self, payload=None, content=None, status_code=200, media_type="image/png"):
        self.status_code = status_code
        self.headers = {
            "Content-Type": "application/json" if content is None else media_type
        }
        self.content = json.dumps(payload).encode() if content is None else content

    @property
    def ok(self):
        """Whether the request succeeded."""
        return self.status_code < 400

    @property
    def text(self):
        """The body of the response, as text."""
        return self.content.decode(errors="replace")

    def json(self):
        """The body of the response, decoded from JSON."""
        return json.loads(self.content)


class FakeSession:
    """
    A stand-in of the REST API of `pettingzoo-server`.

    It serves one agent named `AGENT`, whose observation is the date of the
    step repeated over the observation space, and records the requests it
    received so that the tests may check them.

    Parameters
    ----------
    status_code : int
        The status the endpoints answer, to test the error handling.
    shared : bool
        Whether to answer like a server in the contest mode, which numbers its
        steps in the responses of ``/reset`` and ``/step``.
    render_mode : str or None
        The render mode of the served environment, ``None`` for a server that
        does not render it.

    Attributes
    ----------
    requests : list of tuple
        The ``(method, path, body)`` of each request received.
    headers : dict
        The headers of the session, where the token is expected.
    closed : bool
        Whether the session has been closed.
    render_formats : list of str
        The image format of each request of ``/render``.
    """

    def __init__(self, status_code=200, shared=True, render_mode="rgb_array"):
        self.status_code = status_code
        self.shared = shared
        self.render_mode = render_mode
        self.render_formats = []
        self.requests = []
        self.headers = {}
        self.closed = False
        self.t = 0

    def request(self, method, url, timeout=None, json=None, params=None):
        """Answer a request of the client, as the server would."""
        path = url.split("/api", 1)[1]
        self.requests.append((method, path, json))

        if self.status_code != 200:
            return FakeResponse({"detail": "nope"}, status_code=self.status_code)

        if path == "/env":
            return FakeResponse(
                {
                    "env_id": ENV_ID,
                    "metadata": {"name": "alife-v1", "render_fps": 10},
                    "render_mode": self.render_mode,
                    "possible_agents": [AGENT],
                    "max_num_agents": 1,
                    "observation_spaces": {AGENT: encode_space(OBSERVATION_SPACE)},
                    "action_spaces": {AGENT: encode_space(ACTION_SPACE)},
                }
            )

        if path == "/agents":
            return FakeResponse({"agents": [AGENT], "num_agents": 1})

        if path == "/reset":
            self.t = 0
            return FakeResponse(
                {
                    "agents": [AGENT],
                    "observations": {AGENT: serialization.encode_array(self.observe())},
                    "infos": {AGENT: {"died": False}},
                    "step": self.t if self.shared else None,
                }
            )

        if path == "/step":
            self.t += 1
            return FakeResponse(
                {
                    "agents": [AGENT],
                    "observations": {AGENT: serialization.encode_array(self.observe())},
                    "rewards": {AGENT: 1.5},
                    "terminations": {AGENT: False},
                    "truncations": {AGENT: False},
                    "infos": {AGENT: {"died": True}},
                    "step": self.t if self.shared else None,
                }
            )

        if path == "/render":
            image_format = (params or {}).get("format", "png")
            self.render_formats.append(image_format)
            image = Image.fromarray(np.full((2, 3, 3), self.t, dtype=np.uint8))
            buffer = io.BytesIO()
            image.save(buffer, format=image_format.upper())
            return FakeResponse(content=buffer.getvalue(), media_type=f"image/{image_format}")

        if path == "/state":
            return FakeResponse(
                {"state": serialization.encode_array(np.array([self.t], np.float32))}
            )

        if path == "/close":
            return FakeResponse({"message": "Environment closed successfully"})

        return FakeResponse({"detail": f"Not Found: {path}"}, status_code=404)

    def observe(self):
        """Return the observation of the agent: the date of the step."""
        return np.full(OBSERVATION_SPACE.shape, self.t / 100.0, dtype=np.float32)

    def close(self):
        """Close the session."""
        self.closed = True


def encode_space(space):
    """
    Encode a space as the server does (`app.serialization.encode_space`).

    Only the two spaces of the `alife` environment are supported, which is all
    these tests need.

    Parameters
    ----------
    space : gymnasium.spaces.Space
        The space to encode.

    Returns
    -------
    dict
        The description of the space.
    """
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": space.dtype.str,
            "low": serialization.encode_array(space.low.reshape(-1)[:1]),
            "high": serialization.encode_array(space.high.reshape(-1)[:1]),
        }

    if isinstance(space, spaces.MultiBinary):
        return {"type": "MultiBinary", "n": int(space.n)}

    raise NotImplementedError(f"Unsupported space: {space}")


@pytest.fixture
def session():
    """Return the stand-in of the server."""
    return FakeSession()


@pytest.fixture
def env(session):
    """Return an environment bound to the stand-in of the server."""
    return pzclient.RemoteParallelEnv(
        api_url="http://server/api", token=TOKEN, session=session
    )


###############################################################################
# THE CODEC ###################################################################
###############################################################################


def test_numpy_values_survive_a_round_trip():
    value = {
        "observation": np.arange(6, dtype=np.float32).reshape(2, 3),
        "count": np.int64(3),
        "flags": [np.zeros(2, dtype=np.int8), True, "left"],
    }

    decoded = serialization.decode_value(serialization.encode_value(value))

    assert np.array_equal(decoded["observation"], value["observation"])
    assert decoded["count"] == 3
    assert np.array_equal(decoded["flags"][0], value["flags"][0])
    assert decoded["flags"][1:] == [True, "left"]


def test_the_decoded_arrays_are_writable():
    """Like the arrays of a local environment, e.g. to normalize them in place."""
    array = serialization.decode_value(serialization.encode_value(np.zeros(3)))

    array *= 2.0

    assert array.flags.writeable


@pytest.mark.parametrize(
    "space",
    [
        spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
        spaces.MultiBinary(2),
    ],
)
def test_the_spaces_of_the_server_are_rebuilt(space):
    assert serialization.decode_space(encode_space(space)) == space


def test_an_unsupported_space_is_reported():
    with pytest.raises(NotImplementedError):
        serialization.decode_space({"type": "Graph"})


###############################################################################
# THE PARALLEL API ############################################################
###############################################################################


def test_the_environment_describes_itself(env, session):
    """The description of the environment is read from the server, once."""
    assert session.requests == [("GET", "/env", None)]

    assert env.env_id == ENV_ID
    assert env.metadata["name"] == "alife-v1"
    assert env.render_mode == "rgb_array"
    assert env.possible_agents == [AGENT]
    assert env.max_num_agents == 1
    assert env.observation_space(AGENT) == OBSERVATION_SPACE
    assert env.action_space(AGENT) == ACTION_SPACE

    # No agent is acting before the first reset
    assert env.agents == []
    assert env.num_agents == 0


def test_the_environment_is_a_pettingzoo_parallel_environment(env):
    """A participant may use it wherever a PettingZoo environment is expected."""
    assert isinstance(env, pettingzoo.ParallelEnv)
    assert env.unwrapped is env


def test_the_token_is_sent_in_the_authorization_header(env, session):
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"


def test_the_token_is_required(monkeypatch):
    monkeypatch.delenv("PETTINGZOO_TOKEN", raising=False)

    with pytest.raises(pzclient.AuthenticationError, match="No token"):
        pzclient.RemoteParallelEnv(
            api_url="http://server/api", token="", session=FakeSession()
        )


def test_the_environment_variables_are_read_when_the_environment_is_built(
    monkeypatch, session
):
    """They may be set after `import pzclient`, in a notebook for instance."""
    monkeypatch.setenv("PETTINGZOO_API_URL", "http://notebook/api/")
    monkeypatch.setenv("PETTINGZOO_TOKEN", TOKEN)

    env = pzclient.RemoteParallelEnv(session=session)

    assert env.api_url == "http://notebook/api"
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"


def test_reset_starts_the_episode(env, session):
    observations, infos = env.reset(seed=7)

    assert session.requests[-1] == ("POST", "/reset", {"seed": 7, "options": None})
    assert env.agents == [AGENT]
    assert env.num_agents == 1
    assert env.observation_space(AGENT).contains(observations[AGENT])
    assert infos == {AGENT: {"died": False}}


def test_reset_accepts_a_numpy_seed(env, session):
    """JSON cannot carry a numpy integer as it is: it is sent as a plain int."""
    env.reset(seed=np.int64(7))

    _, _, body = session.requests[-1]
    assert body["seed"] == 7
    assert type(body["seed"]) is int


def test_step_plays_the_actions_of_the_agents(env, session):
    env.reset()

    action = env.action_space(AGENT).sample()
    observations, rewards, terminations, truncations, infos = env.step({AGENT: action})

    # The action travelled as an encoded numpy array
    method, path, body = session.requests[-1]
    assert (method, path) == ("POST", "/step")
    assert np.array_equal(
        serialization.decode_value(body["actions"][AGENT]), action
    )

    assert env.agents == [AGENT]
    assert env.observation_space(AGENT).contains(observations[AGENT])
    assert observations[AGENT].flags.writeable
    assert rewards == {AGENT: 1.5}
    assert terminations == {AGENT: False}
    assert truncations == {AGENT: False}
    assert infos == {AGENT: {"died": True}}


def test_each_action_names_the_step_it_is_meant_for(env, session):
    """The actions carry the step announced by the previous response."""
    env.reset()

    for expected_step in (0, 1, 2):
        env.step({AGENT: env.action_space(AGENT).sample()})
        _, _, body = session.requests[-1]
        assert body["step"] == expected_step


def test_a_training_server_gets_no_step():
    """A server that does not number its steps gets no step with the actions."""
    session = FakeSession(shared=False)
    env = pzclient.RemoteParallelEnv(
        api_url="http://server/api", token=TOKEN, session=session
    )

    env.reset()
    env.step({AGENT: env.action_space(AGENT).sample()})

    _, _, body = session.requests[-1]
    assert body["step"] is None


def test_the_observations_follow_the_steps(env):
    observations, _ = env.reset()
    assert observations[AGENT][0] == pytest.approx(0.0)

    observations, _, _, _, _ = env.step({AGENT: env.action_space(AGENT).sample()})
    assert observations[AGENT][0] == pytest.approx(0.01)


def test_render_returns_the_frame_of_the_environment(env, session):
    env.reset()
    env.step({AGENT: env.action_space(AGENT).sample()})

    frame = env.render()

    # The frames returned are lossless
    assert session.render_formats == ["png"]
    assert frame.shape == (2, 3, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.writeable
    # The stand-in of the server draws the date of the step
    assert np.all(frame == 1)


def test_render_asks_for_nothing_when_the_environment_does_not_render(env, session):
    env.render_mode = None

    assert env.render() is None
    assert env.render_png() is None
    assert ("GET", "/render", None) not in session.requests


def test_nothing_is_rendered_when_the_server_does_not_render():
    session = FakeSession(render_mode=None)

    env = pzclient.RemoteParallelEnv(
        api_url="http://server/api", token=TOKEN, session=session, render_mode="human"
    )

    assert env.render_mode is None
    assert env.render() is None


def test_an_unknown_render_mode_is_refused(session):
    with pytest.raises(ValueError):
        pzclient.RemoteParallelEnv(
            api_url="http://server/api", token=TOKEN, session=session, render_mode="ansi"
        )

    # Refused before anything is asked to the server
    assert session.requests == []


###############################################################################
# THE "HUMAN" RENDER MODE #####################################################
###############################################################################


@pytest.fixture
def pygame(monkeypatch):
    """Return pygame, drawing off-screen so that the tests need no display."""
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    yield pygame

    pygame.display.quit()


@pytest.fixture
def human_env(session, pygame):
    """Return an environment rendered in a window, bound to the stand-in of the server."""
    return pzclient.RemoteParallelEnv(
        api_url="http://server/api", token=TOKEN, session=session, render_mode="human"
    )


def render_requests(session):
    """Return the number of frames the client asked the server for."""
    return session.requests.count(("GET", "/render", None))


def test_the_window_shows_every_reset_and_every_step(human_env, session, pygame):
    human_env.reset()
    assert render_requests(session) == 1

    human_env.step({AGENT: human_env.action_space(AGENT).sample()})
    assert render_requests(session) == 2

    # The window shows the lighter JPEG images
    assert session.render_formats == ["jpeg", "jpeg"]

    # The window opens at the size of the frame, 3 pixels wide and 2 high
    window = pygame.display.get_surface()
    assert window.get_size() == (3, 2)
    # The stand-in of the server draws the date of the step
    assert tuple(window.get_at((0, 0)))[:3] == (1, 1, 1)


def test_render_shows_the_frame_and_returns_nothing(human_env, session, pygame):
    assert human_env.render() is None

    assert render_requests(session) == 1
    assert pygame.display.get_surface() is not None


def test_the_frames_fit_the_resized_window(human_env, pygame):
    human_env.reset()

    # The user enlarges the window: the frame is scaled, keeping its proportions
    pygame.display.set_mode((12, 12), pygame.RESIZABLE)
    human_env.step({AGENT: human_env.action_space(AGENT).sample()})

    window = pygame.display.get_surface()
    assert tuple(window.get_at((6, 6)))[:3] == (1, 1, 1)
    # The frame is 12 x 8 pixels, centered: the top rows are background
    assert tuple(window.get_at((6, 0)))[:3] == (0, 0, 0)


def test_closing_the_window_stops_the_rendering(human_env, session, pygame):
    human_env.reset()

    # The user closes the window
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    human_env.step({AGENT: human_env.action_space(AGENT).sample()})

    assert human_env.render_mode is None
    assert not pygame.display.get_init()

    # The agent keeps playing, and no frame is asked for anymore
    frames = render_requests(session)
    human_env.step({AGENT: human_env.action_space(AGENT).sample()})
    assert render_requests(session) == frames


def test_closing_the_environment_closes_the_window(human_env, pygame):
    human_env.reset()
    assert pygame.display.get_init()

    human_env.close()

    assert not pygame.display.get_init()


def test_state_returns_the_global_state(env):
    env.reset()
    assert np.array_equal(env.state(), np.zeros(1, dtype=np.float32))


def test_remote_agents_reads_the_agents_from_the_server(env, session):
    assert env.remote_agents() == [AGENT]
    assert env.agents == [AGENT]
    assert session.requests[-1] == ("GET", "/agents", None)


def test_close_leaves_the_environment(env, session):
    env.reset()
    env.close()

    assert session.requests[-1] == ("POST", "/close", None)
    assert env.agents == []
    assert session.closed


def test_the_environment_is_a_context_manager(session):
    with pzclient.RemoteParallelEnv(
        api_url="http://server/api", token=TOKEN, session=session
    ) as env:
        env.reset()

    assert session.requests[-1] == ("POST", "/close", None)
    assert session.closed


###############################################################################
# THE ERRORS OF THE SERVER ####################################################
###############################################################################


@pytest.mark.parametrize(
    "status_code, error_class",
    [
        (401, pzclient.AuthenticationError),
        (403, pzclient.PermissionDeniedError),
        (409, pzclient.AgentNotConnectedError),
        (422, pzclient.InvalidActionError),
        (500, pzclient.RemoteEnvError),
        (501, pzclient.RemoteEnvError),
    ],
)
def test_the_errors_of_the_server_are_reported(status_code, error_class):
    """Each failure of the server has its own, catchable, exception."""
    session = FakeSession(status_code=status_code)

    with pytest.raises(error_class) as error:
        pzclient.RemoteParallelEnv(
            api_url="http://server/api", token=TOKEN, session=session
        )

    assert error.value.status_code == status_code
    assert isinstance(error.value, pzclient.RemoteEnvError)


def test_an_unreachable_server_is_reported():
    with pytest.raises(pzclient.ServerUnreachableError):
        pzclient.RemoteParallelEnv(
            api_url="http://localhost:9/api", token=TOKEN, timeout=1.0
        )


def test_an_action_that_cannot_be_encoded_is_invalid(env):
    """A NaN in a plain list is refused before being sent, not blamed on the network."""
    env.reset()

    # A real session: `requests` fails to encode the body before connecting
    # to the (nonexistent) server
    env._session = requests.Session()

    with pytest.raises(pzclient.InvalidActionError, match="cannot be encoded"):
        env.step({AGENT: [float("nan"), 0.0]})


###############################################################################
# THE REGISTRY ################################################################
###############################################################################


def test_the_remote_environments_are_registered():
    """Importing the library registers the environments of the contest."""
    for env_id in ("alife/alife-remote-v1", "remote/parallel-v1"):
        assert env_id in pettingzoo.parallel_registry


def test_the_alife_environment_checks_what_the_server_serves(session):
    """The contest environment refuses to play anything but the `alife` world."""
    env = pzclient.alife_parallel_env(
        api_url="http://server/api", token=TOKEN, session=session
    )
    assert env.env_id == pzclient.ALIFE_ENV_ID

    with pytest.raises(pzclient.RemoteEnvError, match="not 'butterfly/pistonball-v6'"):
        pzclient.RemoteParallelEnv(
            api_url="http://server/api",
            token=TOKEN,
            env_id="butterfly/pistonball-v6",
            session=FakeSession(),
        )
