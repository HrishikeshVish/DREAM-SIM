# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common assertions for type and shape checking."""
import collections.abc
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from planner.type import Array, DType, size_t_any


def assert_collection_of_arrays(arrays: Sequence[Array]) -> None:
    """Checks if ``inputs`` is a collection of arrays.

    .. note::

        Adapted from
        :url:`https://github.com/google-deepmind/chex/blob/master/chex/_src/asserts_internal`.

    Args:
        arrays (Sequence[Array]): The input arrays.

    Raises:
        ValueError: If the input is not a collection of arrays.
    """
    if not isinstance(arrays, collections.abc.Collection):
        raise ValueError(f"Input is not a collection of arrays: {arrays}")


def assert_dtype(
    inputs: Union[Array, Sequence[Array]],
    expected_dtype: Union[DType, Sequence[DType]],
) -> None:
    """Asserts that the input tensor has the expected data type.

    .. note::

        Adapted from
        :url:`https://github.com/google-deepmind/chex/blob/master/chex/_src/asserts.py`

    Args:
        inputs (Union[Array, Sequence[Array]]): The input arrays.
        expected_dtype (Union[DType, Sequence[DType]]): Expected data types.

    Raises:
        AssertionError: If the input arrays do not have the expected data type.
        TypeError: If the input is not an array.
    """
    if not isinstance(inputs, (List, Tuple)):
        inputs = [inputs]
    if not isinstance(expected_dtype, (List, Tuple)):
        expected_dtype = [expected_dtype] * len(inputs)

    err = []
    if len(inputs) != len(expected_dtype):
        raise AssertionError(
            f"Expected {len(inputs)} dtypes, but got {len(expected_dtype)}."
        )
    for idx, (arr, dtype) in enumerate(zip(inputs, expected_dtype)):
        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Expected an array, but got {type(arr)}")
        if arr.dtype != dtype:
            err.append((idx, arr.dtype, dtype))

    if err:
        msg = ";".join(
            [
                f"input {idx} has dtype {dtype}, but expected {expected}"
                for idx, dtype, expected in err
            ]
        )
        raise AssertionError(msg)


def assert_equal_shapes(
    inputs: Sequence[Array],
    *,
    dims: Optional[Union[int, Sequence[int]]] = None,
) -> None:
    """Asserts that the input arrays have the same shape.

    .. note::

        Adapted from
        :url:`https://github.com/google-deepmind/chex/blob/master/chex/_src/asserts.py`

    Args:
        inputs (Sequence[Array]): The input arrays.
        *
        dims (Optional[Union[int, Sequence[int]]]):

    Raises:
        AssertionError: If the input arrays not having the same shape.
    """
    assert_collection_of_arrays(inputs)

    # extract the shapes
    def __extract_shape(
        shape: size_t_any, dims: Optional[Union[int, Sequence[int]]]
    ) -> size_t_any:
        try:
            if dims is None:
                return shape
            elif isinstance(dims, int):
                return shape[dims]
            else:
                return [shape[dim] for dim in dims]
        except IndexError as err:
            raise ValueError(
                f"Encountered indexing error when extracting dims(s) {dims} "
                f"from array shape {shape}"
            ) from err

    expected_shape = [__extract_shape(inputs[0].shape, dims)] * len(inputs)
    shapes = [__extract_shape(arr.shape, dims) for arr in inputs]
    if shapes != expected_shape:
        if dims is not None:
            msg = f"Arrays have different shapes at dims {dims}: {shapes}"
        else:
            msg = f"Arrays have different shapes: {shapes}"
        raise AssertionError(msg)


def assert_shape(
    inputs: Union[Array, Sequence[Array]],
    expected_shape: Union[size_t_any, Sequence[size_t_any]],
) -> None:
    """Asserts that the input arrays have the expected shape.

    .. note::

        Adapted from
        :url:`https://github.com/google-deepmind/chex/blob/master/chex/_src/asserts.py`

    Args:
        inputs (Union[Array, Sequence[Array]): The input arrays.
        expected_shape (Union[size_t_any, Sequence[size_t_any]]): Expected shapes.

    Raises:
        AssertionError: If the input arrays do not have the expected shape.
    """
    if not isinstance(expected_shape, (List, Tuple)):
        raise AssertionError(
            "Expected shape must be a list or tuple, "
            f"but got {expected_shape}."
        )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]
    if not expected_shape or not isinstance(expected_shape[0], (List, Tuple)):
        expected_shape = [expected_shape] * len(inputs)
    if len(inputs) != len(expected_shape):
        raise AssertionError(
            f"Expected {len(inputs)} shapes, but got {len(expected_shape)}."
        )

    err = []
    for idx, (x, shape) in enumerate(zip(inputs, expected_shape)):
        curr_shape = getattr(x, "shape", ())  # by default, assume scalar
        if curr_shape != shape:
            err.append((idx, x.shape, shape))

    if err:
        msg = ";".join(
            [
                f"input {idx} has shape {shape}, but expected {expected}"
                for idx, shape, expected in err
            ]
        )
        raise AssertionError(msg)
