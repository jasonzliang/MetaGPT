
from typing import List, Optional

def longest(strings: List[str]) -> Optional[str]:
    """ Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
    if not strings:  # If the list is empty, return None
        return None

    max_length = 0
    longest_string = ''

    for string in strings:
        if len(string) > max_length:  # If current string length is greater than max_length
            max_length = len(string)  # Update max_length
            longest_string = string  # And update the longest_string

    return longest_string
