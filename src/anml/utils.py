from typing import List


def _check_list_consistency(x, y, Error):
    if not (isinstance(x, List) and isinstance(y, List)):
        raise Error(f"{x} and {y} must be passed in as lists.")
    if not len(x) == len(y):
        raise Error(f"{x.__name__} and {y.__name__} are not of the same length.")
