"""PettingZoo client.

The client library of `pettingzoo-server`: it makes a PettingZoo environment
served over a REST API usable like a local one.

This is the library the participants of the MARL contest install to connect
their agent to the shared `alife` world of the organizers.  Importing it
registers the remote environments in the PettingZoo registry, so that the usual
`pettingzoo.make` builds them:

    >>> import pettingzoo
    >>> import pzclient
    >>> env = pettingzoo.make(                      # doctest: +SKIP
    ...     "parallel",
    ...     "alife/alife-remote-v1",
    ...     api_url="https://https://csc53439ep.jdhp.org/api",
    ...     token="<YOUR_SECRET_TOKEN>",
    ... )

`pzclient.RemoteParallelEnv` may also be instantiated directly, which is the
same thing without the registry.  The ``PETTINGZOO_API_URL`` and
``PETTINGZOO_TOKEN`` environment variables are the equivalents of the
``api_url`` and ``token`` arguments.

Two environment ids are registered:

===========================  ==================================================
Id                           Environment
===========================  ==================================================
``alife/alife-remote-v1``    The `alife` world of the contest; the constructor
                             fails if the server serves anything else.
``remote/parallel-v1``       Whatever parallel environment the server serves.
===========================  ==================================================

Viewing documentation using IPython
-----------------------------------
To see which functions are available in `pzclient`, type ``pzclient.<TAB>``
(where ``<TAB>`` refers to the TAB key).  To view the docstring for a function,
use ``pzclient.get_version?<ENTER>`` (to view the docstring) and
``pzclient.get_version??<ENTER>`` (to view the source code).
"""

import importlib.metadata

# Importing this module registers the remote environments in the PettingZoo
# registry, which is what makes `pettingzoo.make` work with them
from pzclient import envs  # noqa: F401
from pzclient.exceptions import (
    AgentNotConnectedError,
    AuthenticationError,
    InvalidActionError,
    RemoteEnvError,
    ServerUnreachableError,
)
from pzclient.remote_env import (
    ALIFE_ENV_ID,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT,
    RemoteParallelEnv,
    alife_parallel_env,
    parallel_env,
)

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'

__version__ = importlib.metadata.version("pzclient")


def get_version() -> str:
    """
    Return the version of the library.

    Returns
    -------
    str
        The version of `pzclient`.
    """
    return __version__


__all__ = [
    "ALIFE_ENV_ID",
    "DEFAULT_API_URL",
    "DEFAULT_TIMEOUT",
    "AgentNotConnectedError",
    "AuthenticationError",
    "InvalidActionError",
    "RemoteEnvError",
    "RemoteParallelEnv",
    "ServerUnreachableError",
    "alife_parallel_env",
    "get_version",
    "parallel_env",
]
