
from typing import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring.
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
    # Error handling: if the input is not a list, raise an error.
    if not isinstance(strings, list):
        raise ValueError("Input must be a list of strings.")
    
    # Error handling: if the substring is not a string, raise an error.
    if not isinstance(substring, str):
        raise ValueError("Substring must be a string.")
        
    return [s for s in strings if substring in s]
