from typing import List


def _check_list_consistency(x, y, Error):
    if isinstance(x, List) or isinstance(y, List):
        if not (isinstance(x, List) or isinstance(y, List)):
            raise Error(f"{x.__name__} and {y.__name__} are not of the same type.")
        if not len(x) == len(y):
            raise Error(f"{x.__name__} and {y.__name__} are not of the same length.")
