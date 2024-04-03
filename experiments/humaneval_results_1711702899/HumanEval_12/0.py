
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

    longest_string = strings[0]  # Assume the first string is the longest
    for string in strings:  # Iterate over each string in the list
        if len(string) > len(longest_string):  # If this string is longer than the current longest
            longest_string = string  # Update the longest string

    return longest_string  # Return the longest string found
