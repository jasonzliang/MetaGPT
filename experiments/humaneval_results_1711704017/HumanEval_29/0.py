
from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings only for ones that start with a given prefix.
    >>> filter_by_prefix([], 'a')
    []
    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')
    ['abc', 'array']
    """
    # Error handling: if the input is not a list or prefix is not a string, raise an error.
    if not isinstance(strings, list) or not isinstance(prefix, str):
        raise ValueError("Input must be a list of strings and a string respectively.")
    
    return [s for s in strings if s.startswith(prefix)]
