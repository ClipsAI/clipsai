"""
Random utility functions.
"""


def find_missing_dict_keys(data: dict, required_keys: list) -> list:
    """
    Returns a list of keys in 'required_keys' that are missing from 'data'

    Parameters
    ----------
    data: dict
        data to check for missing keys
    required_keys: list
        list of keys that should be in 'data'

    Returns
    -------
    list
        list of keys in 'required_keys' that are missing from 'data'
    """
    missing_keys = []
    for key in required_keys:
        if key not in data.keys():
            missing_keys.append(key)
    return missing_keys
