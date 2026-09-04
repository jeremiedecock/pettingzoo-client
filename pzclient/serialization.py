"""
JSON codec of the numpy objects and of the Gymnasium spaces.

This module is the client half of the codec of the server (the
``app.serialization`` module of `pettingzoo-server`): observations, actions and
states travel as base64 encoded numpy buffers tagged with ``"__ndarray__"``,
and the spaces travel as plain dictionaries describing them.

A remote agent has no access to the sources of the server, which is why the
decoding lives here rather than being imported from it.
"""

import base64
from typing import Any

from gymnasium import spaces
import numpy as np

# Key used by the server to tag the JSON representation of a numpy array
NDARRAY_TAG = "__ndarray__"


def encode_array(array: np.ndarray) -> dict[str, Any]:
    """
    Encode a numpy array into a JSON-serializable dictionary.

    Parameters
    ----------
    array : numpy.ndarray
        The array to encode.

    Returns
    -------
    dict
        A dictionary with the base64 encoded raw buffer of the array, its dtype
        and its shape.
    """
    contiguous_array = np.ascontiguousarray(array)

    return {
        NDARRAY_TAG: base64.b64encode(contiguous_array.tobytes()).decode("ascii"),
        "dtype": contiguous_array.dtype.str,
        "shape": list(contiguous_array.shape),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    """
    Decode a numpy array sent by the server.

    Parameters
    ----------
    payload : dict
        A dictionary with the base64 encoded raw buffer of the array, its dtype
        and its shape.

    Returns
    -------
    numpy.ndarray
        The decoded array.
    """
    buffer = base64.b64decode(payload[NDARRAY_TAG])
    array = np.frombuffer(buffer, dtype=np.dtype(payload["dtype"]))

    return array.reshape(payload["shape"])


def encode_value(value: Any) -> Any:
    """
    Recursively convert a value into a JSON-serializable structure.

    Parameters
    ----------
    value : Any
        The value to encode (numpy array, numpy scalar or container).

    Returns
    -------
    Any
        The JSON-serializable counterpart of `value`.
    """
    if isinstance(value, np.ndarray):
        return encode_array(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]

    return value


def decode_value(value: Any) -> Any:
    """
    Recursively convert a JSON structure received from the server into numpy objects.

    Parameters
    ----------
    value : Any
        The JSON structure to decode.

    Returns
    -------
    Any
        The decoded value.
    """
    if isinstance(value, dict):
        if NDARRAY_TAG in value:
            return decode_array(value)

        return {key: decode_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [decode_value(item) for item in value]

    return value


def decode_space(payload: dict[str, Any]) -> spaces.Space:
    """
    Rebuild a Gymnasium space from the description sent by the server.

    Parameters
    ----------
    payload : dict
        The dictionary describing the space.

    Returns
    -------
    gymnasium.spaces.Space
        The rebuilt space, ready to be sampled.

    Raises
    ------
    NotImplementedError
        If the space type is not supported.
    """
    space_type = payload["type"]

    if space_type == "Box":
        shape = tuple(payload["shape"])
        dtype = np.dtype(payload["dtype"])
        # Uniform bounds are sent as a single value, broadcast them back
        return spaces.Box(
            low=np.broadcast_to(decode_array(payload["low"]), shape).astype(dtype),
            high=np.broadcast_to(decode_array(payload["high"]), shape).astype(dtype),
            shape=shape,
            dtype=dtype,
        )

    if space_type == "Discrete":
        return spaces.Discrete(n=payload["n"], start=payload["start"])

    if space_type == "MultiDiscrete":
        return spaces.MultiDiscrete(
            nvec=decode_array(payload["nvec"]),
            dtype=np.dtype(payload["dtype"]),
            start=decode_array(payload["start"]),
        )

    if space_type == "MultiBinary":
        return spaces.MultiBinary(n=decode_value(payload["n"]))

    if space_type == "Text":
        return spaces.Text(
            max_length=payload["max_length"],
            min_length=payload["min_length"],
            charset=payload["charset"],
        )

    if space_type == "Tuple":
        return spaces.Tuple([decode_space(subspace) for subspace in payload["spaces"]])

    if space_type == "Dict":
        return spaces.Dict(
            {key: decode_space(subspace) for key, subspace in payload["spaces"].items()}
        )

    raise NotImplementedError(f"Unsupported space type: {space_type}")
