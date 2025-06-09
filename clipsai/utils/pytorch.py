"""Utility functions for PyTorch."""

from __future__ import annotations
# standard package imports
import logging
import random

# local package imports
from .exceptions import InvalidComputeDeviceError

# 3rd party imports
try:  # pragma: no cover - optional dependency
    import psutil
except ModuleNotFoundError:  # pragma: no cover - fallback stub
    psutil = None  # type: ignore
try:  # pragma: no cover - optional dependency
    import pynvml
except ModuleNotFoundError:  # pragma: no cover - fallback stub
    pynvml = None  # type: ignore
import torch


def get_valid_torch_devices() -> str or None:
    """
    Returns a list of valid PyTorch devices to use for computation

    Parameters
    ----------
    None

    Returns
    -------
    list
        a list of valid PyTorch devices to use for computation
    """
    return ["cpu", "cuda", "mps"]


def check_valid_torch_device(device: str) -> str or None:
    """
    Checks if the device variable for PyTorch is valid to use for computation on CAI
    backend

    Parameters
    ----------
    device: str
        PyTorch device to perform computations on. Ex: 'cpu', 'cuda'

    Returns
    -------
    str or None
        Returns None if device is valid. Returns a descriptive error message if the
        device is invalid

    Raises
    ------
    InvalidComputeDeviceError: compute device isn't valid and raise_error==True
    """
    valid_devices = get_valid_torch_devices()

    if device not in valid_devices:
        return "Compute device needs to be one of '{}', not '{}'.".format(
            valid_devices, device
        )

    return None


def is_valid_torch_device(device: str) -> bool:
    """
    Returns True if 'device' is a valid PyTorch device to use for computation on CAI
    backend, False otherwise

    Parameters
    ----------
    device: str
        PyTorch device to perform computations on. Ex: 'cpu', 'cuda'

    Returns
    -------
    bool
        True if 'device' is a valid PyTorch device, False otherwise
    """
    msg = check_valid_torch_device(device)
    if msg is None:
        return True
    else:
        return False


def assert_valid_torch_device(device: str) -> bool:
    """
    Raises an error if 'device' is not a valid PyTorch device to use for computation on
    CAI backend

    Parameters
    ----------
    device: str
        PyTorch device to perform computations on. Ex: 'cpu', 'cuda'.

    Returns
    -------
    None

    Raises
    ------
    InvalidComputeDeviceError: device is not a valid PyTorch device to use for
    computation
    """
    msg = check_valid_torch_device(device)
    if msg is not None:
        raise InvalidComputeDeviceError(msg)


def get_compute_device() -> str:
    """
    Returns the compute device to use for computation.

    Parameters
    ----------
    None

    Returns
    -------
    str
        PyTorch device to perform computations on. Ex: 'cpu', 'cuda'
    """
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def check_compute_device_available(device: str) -> str or None:
    """
    Checks if 'device' is valid and is available for performing
    computation on the current machine.

    Parameters
    ----------
    device: str
        Compute device to perform computations on. Ex: 'cpu', 'cuda'.

    Returns
    -------
    str or None
        Returns None if 'device' is valid for CAI backend and that is available for
        performing computation on the current machine. Returns a descriptive error
        message if not.
    """
    msg = check_valid_torch_device(device)
    if msg is not None:
        return msg

    # check if 'cuda' is available
    if device == "cuda" and torch.cuda.is_available() is False:
        return "Device 'cuda' is not available for computation on this machine"

    # check if 'mps' isi available
    if device == "mps" and torch.backends.mps.is_available() is False:
        return "Device 'mps' is not available for computation on this machine"

    # 'cpu' is always available
    return None


def is_compute_device_available(device: str) -> bool:
    """
    Returns True if 'device' is valid for CAI backend and is available for performing
    computation on the current machine, False otherwise.

    Parameters
    ----------
    device: str
        Compute device to perform computations on. Ex: 'cpu', 'cuda'.

    Returns
    -------
    bool
        True if 'device' is valid for CAI backend and is available for performing
        computation on the current machine, False otherwise.
    """
    return check_compute_device_available(device) is None


def assert_compute_device_available(device: str) -> None:
    """
    Raises an error if 'device' is not valid for CAI backend or is not available for
    performing computation on the current machine.

    Parameters
    ----------
    device: str
        Compute device to perform computation on

    Returns
    -------
    None

    Raises
    ------
    InvalidComputeDeviceError: compute device isn't valid or isn't available
    """
    error = check_compute_device_available(device)
    if error is not None:
        raise InvalidComputeDeviceError(error)


def max_magnitude_2d(tensor: torch.tensor, dim: int) -> torch.tensor:
    """
    Returns the maximum magnitude of values in a tensor along dimension 1 (max value in
    each row) or dimension 0 (max value in each column).

    Parameters
    ----------
    tensor: torch.tensor
        2 dimensional tensor
    dim: int
        dimension to perform max operation across;
        must be dim 0 or dim 1

    Returns
    -------
    max_tensor: torch.tensor
        1 dimensional tensor containing the max value along given dimensions
    """
    if torch.is_tensor(tensor) is False:
        msg = "tensor must be of type 'torch.Tensor' not {}".format(type(tensor))
        logging.error(msg)
        raise TypeError(msg)
    if isinstance(dim, int) is False:
        msg = "dim must be of type 'int' not {}".format(type(tensor))
        logging.error(msg)
        raise TypeError(msg)

    if dim not in [0, 1]:  # dimension size incompatible with this function
        raise ValueError("dim must be '0' or '1', not {}".format(dim))

    positive_tensor = torch.abs(tensor)
    _, max_idcs = torch.max(positive_tensor, dim)

    dim0, dim1 = tensor.shape

    # perform over dimension 0 --> max value in each column
    if dim == 0:
        max_tensor = torch.empty(dim1)
        max_tensor = tensor[max_idcs, range(dim1)]
    # perform over dimension 1 --> max value in each row
    elif dim == 1:
        max_tensor = torch.empty(dim0)
        max_tensor = tensor[range(dim0), max_idcs]

    return max_tensor


def reset_seed(number):
    """
    Reset random seed to the specific number

    Parameters
    ----------
    None

    Returns
    -------
    number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)


def mem_stats() -> dict:
    """
    Returns a dictionary containing the memory statistics for the given device.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        dictionary containing the memory statistics for GPU and CPU. The dictionary
        contains the following keys:
            - "gpu": dictionary containing the following keys:
                - "total": total GPU memory
                - "free": free GPU memory
            - "cpu": dictionary containing the following keys:
                - "total": total CPU memory
                - "free": free CPU memory
    """
    # get the free gpu memory
    total_gpu_memory = 0
    free_gpu_memory = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        total_gpu_memory = info.total
        free_gpu_memory = info.free

    # get free cpu memory
    total_cpu_memory = psutil.virtual_memory().total
    free_cpu_memory = psutil.virtual_memory().available

    return {
        "gpu": {"total": total_gpu_memory, "free": free_gpu_memory},
        "cpu": {"total": total_cpu_memory, "free": free_cpu_memory},
    }


def get_free_cpu_memory() -> int:
    """
    Returns the free CPU memory in bytes.

    Parameters
    ----------
    None

    Returns
    -------
    free_memory: int
        The free CPU memory in bytes.
    """
    return psutil.virtual_memory().available
