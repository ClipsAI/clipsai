"""
Utilities functions for PyTorch not requiring ML dependencies.
"""
# local package imports
from .exceptions import InvalidComputeDeviceError


def get_valid_torch_devices() -> str or None:
    """
    Returns a list of valid PyTorch devices to use for computation on CAI backend

    Parameters
    ----------
    None

    Returns
    -------
    list
        a list of valid PyTorch devices to use for computation on CAI backend
    """
    return ["cpu", "cuda", "mps"]


def check_valid_torch_device(device: str) -> str or None:
    """
    Checks if the device variable for PyTorch is valid to use for computation on CAI
    backend

    Parameters
    ----------
    device: str
        The PyTorch device to perform computation on

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
        the pytorch hardware device to use for computation

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
        the pytorch hardware device to use for computation

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
