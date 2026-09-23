"""
The PettingZoo parallel environments served by `pettingzoo-server`.

`RemoteParallelEnv` implements ``pettingzoo.ParallelEnv`` on top of the REST
API of the server: every call of the parallel API is an HTTP request, the
environment itself runs on the server and the agent runs here.  Apart from the
two constructor arguments (the URL of the API and the token of the
participant), it is used exactly like a local PettingZoo environment::

    import pettingzoo
    import pzclient  # registers the remote environments

    env = pettingzoo.make("parallel", "alife/alife-remote-v1", token="<YOUR_SECRET_TOKEN>")

    observations, infos = env.reset()

    for _ in range(1000):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()

The mapping between the parallel API and the endpoints of the server is:

===============================  ==========================
`RemoteParallelEnv`              Endpoint
===============================  ==========================
``__init__``                     ``GET /env``
``agents`` / ``num_agents``      ``GET /agents``
``reset``                        ``POST /reset``
``step``                         ``POST /step``
``render``                       ``GET /render``
``state``                        ``GET /state``
``close``                        ``POST /close``
===============================  ==========================

The server always renders the environment as images; ``render_mode`` only
chooses what the client does with them: ``"rgb_array"`` (the default) returns
them as numpy frames, decoded from lossless PNG images, e.g. to show them in a
notebook, and ``"human"`` shows them in a pygame window (c.f.
`pzclient.viewer`), refreshed after every `reset` and `step`.  The window shows
JPEG images, which the server encodes ~20 times faster than PNG ones and which
are ~5 times smaller.
"""

import io
import logging
import os
from typing import Any, Self

from gymnasium import spaces
import numpy as np
import pettingzoo
from PIL import Image
import requests

from pzclient import serialization, viewer
from pzclient.exceptions import (
    AgentNotConnectedError,
    AuthenticationError,
    InvalidActionError,
    PermissionDeniedError,
    RemoteEnvError,
    ServerUnreachableError,
)

# Configure logging
logger = logging.getLogger(__name__)

# Base URL of the API of the server, used when neither the `api_url` argument
# nor the "PETTINGZOO_API_URL" environment variable is given.  The environment
# variables ("PETTINGZOO_API_URL" and "PETTINGZOO_TOKEN") are read when the
# environment is built, not when the module is imported, so that they may be
# set after `import pzclient` (in a notebook, for instance).
DEFAULT_API_URL = "http://localhost:8000/api"

# Timeout of the HTTP requests, in seconds.  It must be larger than the step
# timeout of the server: in the contest mode a step waits for the actions of
# all the connected participants before answering.
DEFAULT_TIMEOUT = 60.0

# The id of the "alife" environment of the contest, in the registry of the
# server (`alife_parallel_env` refuses to play anything else)
ALIFE_ENV_ID = "alife/alife-v1"

# The render modes of the client: the server always sends PNG images, which are
# either returned as numpy frames or shown in a window
RENDER_MODES = ("human", "rgb_array")


class RemoteParallelEnv(pettingzoo.ParallelEnv):
    """
    A PettingZoo parallel environment served by `pettingzoo-server`.

    The instance behaves like the environment it proxies: agent ids are
    strings, observations and actions are numpy objects and the spaces are the
    ones of the remote environment.

    The server serves the environment in one of two modes, which the
    participant does not choose:

    - in the *training* mode, the environment is private: `possible_agents`
      holds all its agents and the participant drives them all;
    - in the *contest* mode, a single environment is shared by all the
      participants and each of them controls exactly one agent of it, named
      after the participant: `possible_agents` holds that single agent, `reset`
      connects it to the shared world and `close` disconnects it.  The
      participants come and go at any moment of an eternal episode, so
      ``while env.agents:`` never ends: bound the loop of your agent.  The
      world plays its steps on its own clock, and ``infos[agent]
      ["shared_world"]`` tells whether your action was played (see `step`).

    Parameters
    ----------
    api_url : str, optional
        The base URL of the API, e.g. ``"http://localhost:8000/api"``; defaults
        to the ``PETTINGZOO_API_URL`` environment variable, else to
        `DEFAULT_API_URL`.
    token : str, optional
        The token identifying the participant, given by the organizers of the
        contest; defaults to the ``PETTINGZOO_TOKEN`` environment variable.
    timeout : float
        The timeout of the HTTP requests, in seconds.  It must be larger than
        the step timeout of the server.
    env_id : str, optional
        The id of the environment the server is expected to serve; the
        constructor fails if the server serves another one.
    session : requests.Session, optional
        The HTTP session to use; a new one is created by default.
    render_mode : str or None
        ``"rgb_array"`` to get the frames of `render` as numpy arrays,
        ``"human"`` to show them in a pygame window refreshed after every
        `reset` and `step` (pygame must be installed), ``None`` to disable
        rendering.  It is ``None`` whatever is asked if the server does not
        render the environment.  Rendering is reserved to the organizers: for
        a participant, `render` raises `pzclient.PermissionDeniedError`.

    Attributes
    ----------
    agents : list of str
        The agents currently acting, updated by `reset` and `step`, empty
        before the first `reset`.
    possible_agents : list of str
        All the agents the participant may control.
    env_id : str
        The id of the environment served by the server.
    metadata : dict
        The metadata of the remote environment.
    render_mode : str or None
        The render mode of the environment (see the `render_mode` parameter).
    observation_spaces : dict
        The observation space of each agent of `possible_agents`.
    action_spaces : dict
        The action space of each agent of `possible_agents`.

    Raises
    ------
    ValueError
        If `render_mode` is not one of `RENDER_MODES` or ``None``.
    ImportError
        If `render_mode` is ``"human"`` and pygame is not installed.
    pzclient.AuthenticationError
        If no token is given, or if the server refuses it.
    pzclient.ServerUnreachableError
        If the server cannot be reached.
    pzclient.RemoteEnvError
        If the server does not serve the expected environment.
    """

    def __init__(
        self,
        api_url: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        env_id: str | None = None,
        session: requests.Session | None = None,
        render_mode: str | None = "rgb_array",
    ):
        if render_mode is not None and render_mode not in RENDER_MODES:
            raise ValueError(
                f"Unknown render mode '{render_mode}', expected one of "
                f"{RENDER_MODES} or None."
            )

        if render_mode == "human":
            # Fail now rather than at the first frame if pygame is missing
            viewer.import_pygame()

        # The window of the "human" render mode, opened by the first frame
        self._viewer: viewer.Viewer | None = None

        self.api_url = (
            api_url or os.getenv("PETTINGZOO_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self.timeout = timeout

        token = token or os.getenv("PETTINGZOO_TOKEN")

        if not token:
            raise AuthenticationError(
                "No token: pass 'token=...' to the environment or set the "
                "'PETTINGZOO_TOKEN' environment variable with the token given "
                "to you by the organizers of the contest."
            )

        self._session = session if session is not None else requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {token}"})

        # No agent is acting before the first reset
        self.agents: list[str] = []

        # The step the next actions are meant for, as numbered by the server
        # in the contest mode (``None`` in the training mode, or before the
        # first reset): it is sent with the actions, so that the server ignores
        # them if the world has moved on in the meantime
        self._step: int | None = None

        description = self._request("GET", "/env")

        self.env_id: str = description["env_id"]

        if env_id is not None and self.env_id != env_id:
            self._session.close()
            raise RemoteEnvError(
                f"The server at {self.api_url} serves '{self.env_id}', "
                f"not '{env_id}'."
            )

        self.metadata: dict[str, Any] = description["metadata"]
        # The server does not render the environment: there is nothing to show
        if description["render_mode"] is None:
            if render_mode == "human":
                logger.warning(
                    f"The server at {self.api_url} does not render "
                    f"'{self.env_id}': the 'human' render mode is disabled."
                )
            render_mode = None

        self.render_mode: str | None = render_mode
        self.possible_agents: list[str] = description["possible_agents"]

        self.observation_spaces: dict[str, spaces.Space] = {
            agent: serialization.decode_space(space)
            for agent, space in description["observation_spaces"].items()
        }
        self.action_spaces: dict[str, spaces.Space] = {
            agent: serialization.decode_space(space)
            for agent, space in description["action_spaces"].items()
        }

        logger.info(
            f"Connected to '{self.env_id}' on {self.api_url} "
            f"as {', '.join(self.possible_agents)}"
        )

    ###########################################################################
    # THE PARALLEL API ########################################################
    ###########################################################################

    def observation_space(self, agent: str) -> spaces.Space:
        """
        Return the observation space of an agent.

        Parameters
        ----------
        agent : str
            The id of the agent.

        Returns
        -------
        gymnasium.spaces.Space
            The observation space of `agent`.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """
        Return the action space of an agent.

        Parameters
        ----------
        agent : str
            The id of the agent.

        Returns
        -------
        gymnasium.spaces.Space
            The action space of `agent`.
        """
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Start (or restart) the episode and return the first observations.

        In the contest mode the world is shared with the other participants and
        is therefore never reset: this call connects a brand new agent of yours
        to the running world, and `seed` and `options` are ignored by the
        server.

        Parameters
        ----------
        seed : int, optional
            The seed of the episode.
        options : dict, optional
            The reset options of the environment.

        Returns
        -------
        tuple of dict
            The initial observations and infos of the agents.
        """
        # The seed and the options may hold numpy objects (``np.int64`` seeds,
        # for instance), which JSON cannot carry as they are
        response = self._request(
            "POST",
            "/reset",
            json=serialization.encode_value({"seed": seed, "options": options}),
        )

        self.agents = response["agents"]
        self._step = response.get("step")

        if self.render_mode == "human":
            self.render()

        return (
            serialization.decode_value(response["observations"]),
            serialization.decode_value(response["infos"]),
        )

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """
        Play one parallel step with exactly one action per acting agent.

        In the contest mode, the shared world plays its steps on its own clock:
        the call returns once the world has played the step, which happens as
        soon as every connected participant has sent its action, and at the
        latest once the step timeout of the server has expired (an agent that
        answers too late does nothing during that step).  An action that
        arrives after its step has been played is ignored: the call then
        returns at once, with the latest observation of your agent.
        ``infos[agent]["shared_world"]`` tells what became of the action:
        ``action_applied`` (whether it was played), ``message`` (why it was
        not) and ``step`` (the number of the next step of the world).

        Parameters
        ----------
        actions : dict
            The actions of the agents, keyed by agent id.  Exactly one action
            per agent of `agents` is expected.

        Returns
        -------
        tuple of dict
            The observations, rewards, terminations, truncations and infos of
            the agents.

        Raises
        ------
        pzclient.AgentNotConnectedError
            If no agent of yours is acting: call `reset` first.
        pzclient.InvalidActionError
            If the actions do not match the acting agents or their action
            spaces.
        pzclient.RemoteEnvError
            If the shared world did not play the step in due time (HTTP 503,
            its ``status_code``): try again.
        """
        response = self._request(
            "POST",
            "/step",
            json={"actions": serialization.encode_value(actions), "step": self._step},
        )

        self.agents = response["agents"]
        self._step = response.get("step")

        if self.render_mode == "human":
            self.render()

        return (
            serialization.decode_value(response["observations"]),
            response["rewards"],
            response["terminations"],
            response["truncations"],
            serialization.decode_value(response["infos"]),
        )

    def render(self) -> np.ndarray | None:
        """
        Render the current state of the remote environment.

        In the contest mode, this is a picture of the whole shared world.

        In the ``"human"`` render mode, the frame is shown in a pygame window,
        opened by the first frame, and nothing is returned.  Closing that
        window stops the rendering (`render_mode` becomes ``None``) while the
        agents keep playing; set `render_mode` back to ``"human"`` to reopen it.

        Returns
        -------
        numpy.ndarray or None
            The ``(H, W, 3)`` uint8 frame of the current state in the
            ``"rgb_array"`` render mode, as it does locally; ``None`` in the
            ``"human"`` render mode, or if the environment does not render
            anything.

        Raises
        ------
        pzclient.PermissionDeniedError
            Unless the participant is an administrator: rendering is reserved
            to the organizers, in the training mode as in the contest mode
            (where the picture of the world reveals where the agents of the
            other participants are and what they do).
        """
        if self.render_mode == "human":
            content = self._render_image("jpeg")

            if content is not None:
                if self._viewer is None:
                    self._viewer = viewer.Viewer(f"{self.env_id} on {self.api_url}")

                if not self._viewer.show(content):
                    logger.info("The render window was closed: rendering stopped.")
                    self.render_mode = None

            return None

        content = self.render_png()

        if content is None:
            return None

        # `np.array` rather than `np.asarray`, which would return a read-only
        # view of the pixels of the image
        return np.array(Image.open(io.BytesIO(content)))

    def state(self) -> Any:
        """
        Return the global state of the remote environment.

        Returns
        -------
        Any
            The global state, as returned by ``ParallelEnv.state``.

        Raises
        ------
        pzclient.PermissionDeniedError
            In the contest mode, unless the participant is an administrator:
            the global state reveals what the agents of the other participants
            sense and do.
        pzclient.RemoteEnvError
            If the remote environment does not implement a global state.
        """
        return serialization.decode_value(self._request("GET", "/state")["state"])

    def close(self) -> None:
        """
        Leave the environment and release the HTTP session.

        In the contest mode, the agent of the participant leaves the shared
        world, which keeps living for the other participants; calling `reset`
        connects a new agent to it.  The window of the ``"human"`` render mode
        is closed.
        """
        try:
            self._request("POST", "/close")
        except RemoteEnvError as error:
            logger.warning(f"The environment could not be closed cleanly: {error}")
        finally:
            self.agents = []
            self._session.close()

            if self._viewer is not None:
                self._viewer.close()

    ###########################################################################
    # EXTRAS ##################################################################
    ###########################################################################

    def render_png(self) -> bytes | None:
        """
        Render the current state of the remote environment as a PNG image.

        This is what the server actually sends; `render` decodes it into the
        numpy frame of the PettingZoo API.

        Returns
        -------
        bytes or None
            The PNG encoded frame, or ``None`` if the environment does not
            render anything (its ``render_mode`` is ``None``).

        Raises
        ------
        pzclient.PermissionDeniedError
            Unless the participant is an administrator (see `render`).
        """
        return self._render_image("png")

    def _render_image(self, image_format: str) -> bytes | None:
        """
        Ask the server for an image of the current state of the environment.

        Parameters
        ----------
        image_format : {"png", "jpeg"}
            The format of the image: PNG is lossless, JPEG is much faster to
            encode and much smaller.

        Returns
        -------
        bytes or None
            The encoded frame, or ``None`` if the environment does not render
            anything (its ``render_mode`` is ``None``).
        """
        if self.render_mode is None:
            return None

        content = self._request("GET", "/render", params={"format": image_format})

        return content if isinstance(content, bytes) else None

    def remote_agents(self) -> list[str]:
        """
        Ask the server which agents of the participant are acting.

        `agents` is the local copy of that list, updated by `reset` and `step`;
        this method reads it from the server, which is useful after a network
        error.

        Returns
        -------
        list of str
            The acting agents.
        """
        self.agents = self._request("GET", "/agents")["agents"]

        return self.agents

    def __enter__(self) -> Self:
        """Return the environment, to be used as a context manager."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Close the environment when leaving the ``with`` block."""
        self.close()

    ###########################################################################
    # HTTP ####################################################################
    ###########################################################################

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """
        Send a request to the API and return its decoded response.

        Parameters
        ----------
        method : str
            The HTTP method of the request.
        path : str
            The path of the endpoint, relative to the base URL.
        **kwargs
            The extra arguments passed to ``requests.Session.request``.

        Returns
        -------
        Any
            The JSON body of the response, or its raw content for the binary
            responses (the PNG image of ``/render``).

        Raises
        ------
        pzclient.InvalidActionError
            If the body of the request cannot be encoded as JSON.
        pzclient.ServerUnreachableError
            If the server cannot be reached.
        pzclient.RemoteEnvError
            If the server answers with an error status; the subclass of the
            error tells what happened (see `pzclient.exceptions`).
        """
        try:
            response = self._session.request(
                method, f"{self.api_url}{path}", timeout=self.timeout, **kwargs
            )
        except requests.exceptions.InvalidJSONError as error:
            # Raised while the body is encoded, before anything is sent (JSON
            # has no NaN, for instance): the request is invalid, like the ones
            # the server rejects with a 422, and the server is not to blame
            raise InvalidActionError(
                f"{method} {path} cannot be encoded as JSON: {error}"
            )
        except requests.RequestException as error:
            raise ServerUnreachableError(
                f"{method} {path} could not reach the server at {self.api_url}: {error}"
            )

        if not response.ok:
            raise self._error(method, path, response)

        if response.headers.get("Content-Type", "").startswith("application/json"):
            return response.json()

        return response.content

    @staticmethod
    def _error(
        method: str, path: str, response: requests.Response
    ) -> RemoteEnvError:
        """
        Build the error describing a failed response of the server.

        Parameters
        ----------
        method : str
            The HTTP method of the request.
        path : str
            The path of the endpoint.
        response : requests.Response
            The failed response.

        Returns
        -------
        pzclient.RemoteEnvError
            The error to raise.
        """
        # FastAPI reports the cause of the error in a "detail" field
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        error_classes = {
            401: AuthenticationError,
            403: PermissionDeniedError,
            409: AgentNotConnectedError,
            422: InvalidActionError,
        }
        error_class = error_classes.get(response.status_code, RemoteEnvError)

        return error_class(
            f"{method} {path} failed [{response.status_code}]: {detail}",
            status_code=response.status_code,
        )


def parallel_env(**kwargs: Any) -> RemoteParallelEnv:
    """
    Instantiate the environment served by `pettingzoo-server`, whatever it is.

    This factory is the entry point registered as ``"remote/parallel-v1"`` in
    the PettingZoo registry, and the ``parallel_env`` name expected from a
    PettingZoo environment module.

    Parameters
    ----------
    **kwargs
        The keyword arguments of `RemoteParallelEnv`.

    Returns
    -------
    RemoteParallelEnv
        The environment, ready to be reset.
    """
    return RemoteParallelEnv(**kwargs)


def alife_parallel_env(**kwargs: Any) -> RemoteParallelEnv:
    """
    Instantiate the `alife` environment of the contest, served remotely.

    This factory is the entry point registered as ``"alife/alife-remote-v1"``
    in the PettingZoo registry.  It is `parallel_env` with a check: the server
    must really serve the `alife` environment of the contest.

    Parameters
    ----------
    **kwargs
        The keyword arguments of `RemoteParallelEnv`.

    Returns
    -------
    RemoteParallelEnv
        The environment, ready to be reset.

    Raises
    ------
    pzclient.RemoteEnvError
        If the server serves another environment.
    """
    kwargs.setdefault("env_id", ALIFE_ENV_ID)

    return RemoteParallelEnv(**kwargs)
