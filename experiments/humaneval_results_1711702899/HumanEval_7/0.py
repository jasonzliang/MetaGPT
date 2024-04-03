
from typing import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """Filter an input list of strings only for ones that contain given substring.

    Args:
        strings: A list of strings to be filtered.
        substring: The substring to filter the strings by.

    Returns:
        A new list containing only those strings from the input list that contain the specified substring.

    Raises:
        TypeError: If 'strings' is not a list or if 'substring' is not a string.
    """
    # Checking if inputs are of correct type
    if not isinstance(strings, list) or not all(isinstance(s, str) for s in strings):
        raise TypeError("'strings' must be a list of strings.")
    if not isinstance(substring, str):
        raise TypeError("'substring' must be a string.")
    
    # Filtering the input list
    filtered_list = [s for s in strings if substring in s]
    
    return filtered_list
