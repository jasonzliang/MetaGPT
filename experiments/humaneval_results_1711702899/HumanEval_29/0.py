
from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """Filter an input list of strings only for ones that start with a given prefix.

    Args:
        strings: A list of strings to be filtered.
        prefix: The prefix to filter the strings by.

    Returns:
        A new list containing only the strings from the input list that start with the specified prefix.

    Raises:
        TypeError: If 'strings' is not a list or 'prefix' is not a string.
    """
    if not isinstance(strings, list) or not all(isinstance(s, str) for s in strings):
        raise TypeError("'strings' must be a list of strings.")
    if not isinstance(prefix, str):
        raise TypeError("'prefix' must be a string.")
    
    return [s for s in strings if s.startswith(prefix)]
