"""
Registration of the remote environments in the PettingZoo registry.

PettingZoo keeps its environments in two registries (``aec_registry`` and
``parallel_registry``) filled with `pettingzoo.register`, and instantiates them
with `pettingzoo.make`.  This is the standard mechanism used by the official
environments, and custom environments join the same registries the same way
(https://pettingzoo.farama.org/tutorials/custom_environment/).

Importing `pzclient` registers the environments served by
`pettingzoo-server`, which are then instantiated exactly like a local one::

    import pettingzoo
    import pzclient

    env = pettingzoo.make("parallel", "alife/alife-remote-v1", token="token_abc123")

The entry points are declared as ``"module:factory"`` strings rather than as
callables, so that nothing is imported until an environment is actually
instantiated.
"""

import pettingzoo

# The environments added to `pettingzoo.parallel_registry`, mapping their
# PettingZoo id to the factory instantiating them
PARALLEL_ENTRY_POINTS: dict[str, str] = {
    # The "alife" artificial-life environment of the contest, served remotely
    # (https://github.com/jeremiedecock/alife)
    "alife/alife-remote-v1": "pzclient.remote_env:alife_parallel_env",
    # Whatever parallel environment the server serves, e.g. an official
    # PettingZoo environment used to test an agent
    "remote/parallel-v1": "pzclient.remote_env:parallel_env",
}

for env_id, entry_point in PARALLEL_ENTRY_POINTS.items():
    pettingzoo.register("parallel", env_id, entry_point=entry_point)
