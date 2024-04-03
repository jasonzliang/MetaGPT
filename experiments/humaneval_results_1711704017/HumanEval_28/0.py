
from typing import List

def concatenate_strings(strings: List[str]) -> str:
    """Concatenates a list of strings into a single string.

    Args:
        strings (List[str]): A list of strings to be concatenated.

    Returns:
        str: The resulting concatenated string.

    Raises:
        TypeError: If the input is not a list or if it contains non-string elements.

    Example:
        >>> concatenate_strings(['a', 'b', 'c'])
        'abc'
    """
    # Checking if the input is a list of strings
    if not isinstance(strings, list) or not all(isinstance(s, str) for s in strings):
        raise TypeError("Input must be a list of strings.")
    
    return ''.join(strings)
