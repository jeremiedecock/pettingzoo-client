"""
The exceptions raised by `pzclient`.

They all derive from `RemoteEnvError`, so a participant may catch everything
this library raises with a single ``except pzclient.RemoteEnvError``.
"""


class RemoteEnvError(Exception):
    """
    Base class of the errors of the remote environment.

    Parameters
    ----------
    message : str
        The description of the error.
    status_code : int, optional
        The HTTP status returned by the server, when the error comes from an
        answer of the server.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(RemoteEnvError):
    """The token is missing, unknown or refused by the server (HTTP 401/403)."""


class AgentNotConnectedError(RemoteEnvError):
    """
    No agent is acting: `reset` has to be called first (HTTP 409).

    In the contest mode, this is also raised when the agent left the shared
    world -- because it was idle for too long, for instance.  Calling `reset`
    connects a new agent to the world.
    """


class InvalidActionError(RemoteEnvError):
    """The actions do not match the acting agents or their action spaces (HTTP 422)."""


class ServerUnreachableError(RemoteEnvError):
    """The server could not be reached at all (network error, timeout, ...)."""
