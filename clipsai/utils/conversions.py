"""
Unit conversion functions.
"""
# standard library imports
import math

# time
SECS_PER_SEC = 1
SECS_PER_MIN = 60
SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400

# SI prefixes
GIGA = 10**9
NANO = 10 ** (-9)
GIBI = 1024**3

SECS_IDX = 0
MINUTES_IDX = 1
HOURS_IDX = 2
DAYS_IDX = 3


def seconds_to_hms_time_format(seconds: float, num_digits: int = 3) -> str:
    """
    Converts number of seconds into a string format of
    '{hours}:{minutes}:{seconds}.{num_digits}' rounded to three decimal places by
    default

    Parameters
    ----------
    seconds: float
        number of seconds
    num_digits: int
        number of decimal places to round to

    Returns
    -------
    hour_format: str
        time formatted to the hour as a string
    """
    if num_digits < 0:
        raise ValueError("num_digits ({}) cannot be negative".format(num_digits))
    is_negative = seconds < 0
    seconds = abs(seconds)

    hours = int(seconds // 3600)
    hours_rmdr = seconds % 3600
    mins = int(hours_rmdr // 60)
    secs = round(hours_rmdr % 60, num_digits)

    # if num_digits == 0 --> lose decimal place and decimal --> width drops by two
    secs_tot_width = 3 + num_digits - (num_digits == 0)
    hour_format = f"{hours:02d}:{mins:02d}:{secs:0{secs_tot_width}.{num_digits}f}"
    if is_negative and ((hours + mins + secs) != 0):
        hour_format = "-" + hour_format

    return hour_format


def hms_time_format_to_seconds(hms_time: str) -> float:
    """
    Converts a string in the HMS time format of '{hours}:{minutes}:{seconds}' to seconds

    Parameters
    ----------
    hms_time: str
        time in the HMS format of '{hours}:{minutes}:{seconds}'

    Returns
    -------
    seconds: float
        time formatted to the hour as a string
    """
    hms_times = hms_time.strip().split(":")

    hms_times = [float(x) for x in hms_times]
    hms_times.reverse()
    conversions_to_seconds = {
        SECS_IDX: SECS_PER_SEC,
        MINUTES_IDX: SECS_PER_MIN,
        HOURS_IDX: SECS_PER_HOUR,
        DAYS_IDX: SECS_PER_DAY,
    }
    seconds = 0
    for i in range(len(hms_times)):
        seconds += hms_times[i] * conversions_to_seconds[i]

    return seconds


def hours_to_seconds(hours: float) -> float:
    """
    Converts hours to seconds

    Parameters
    ----------
    hours: float
        number of hours

    Returns
    -------
    seconds: float
        number of seconds in hours
    """
    return hours * SECS_PER_HOUR


def seconds_to_hours(seconds: float) -> float:
    """
    Converts seconds to hours

    Parameters
    ----------
    seconds: float
        number of seconds

    Returns
    -------
    hours: float
        number of hours in seconds
    """
    return seconds / SECS_PER_HOUR


def bytes_to_gigabytes(bytes: int) -> float or int:
    """
    Converts bytes to gigabytes

    Parameters
    ----------
    bytes: int
        the number of bytes to convert to gigabytes

    Returns
    -------
    gigabytes: float or int
    """
    return bytes / GIGA


def gigabytes_to_bytes(gigabytes: float or int) -> int:
    """
    Converts gigabytes to bytes

    - If the precision of gigabytes exceeds 9 decimal places, bytes is rounded up to
    the nearest integer

    Parameters
    ----------
    gigabytes: float or int
        the number of gigabytes to convert to bytes

    Returns
    -------
    bytes: int
        the number of bytes
    """
    return math.ceil(gigabytes * GIGA)


def secs_to_nanosecs(seconds: float) -> int:
    """
    Converts seconds to nanoseconds

    Parameters
    ----------
    seconds: float
        number of seconds

    Returns
    -------
    nanoseconds: int
        number of nanoseconds in seconds
    """
    return int(seconds / NANO)


def nano_secs_to_secs(nano_secs: int) -> float:
    """
    Converts nanoseconds to seconds

    Parameters
    ----------
    nano_secs: int
        number of nanoseconds

    Returns
    -------
    seconds: float
        number of seconds in nanoseconds
    """
    return nano_secs * NANO


def bytes_to_gibibytes(bytes: int) -> float:
    """
    Converts bytes to gibibytes

    Parameters
    ----------
    bytes: int
        the number of bytes to convert to gibibytes

    Returns
    -------
    gibibytes: float
    """
    return bytes / GIBI


def gibibytes_to_bytes(gibibytes: float) -> int:
    """
    Converts gibibytes to bytes

    - If the precision of gibibytes exceeds 9 decimal places, bytes is rounded up to
    the nearest integer

    Parameters
    ----------
    gibibytes: float
        the number of gibibytes to convert to bytes

    Returns
    -------
    bytes: int
        the number of bytes
    """
    return math.ceil(gibibytes * GIBI)
