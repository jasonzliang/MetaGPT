
from typing import List

def concatenate_strings(string_list: List[str]) -> str:
    """Concatenates a list of strings into a single string.

    Args:
        string_list: A list of strings to be concatenated.

    Returns:
        The resulting string after all input strings have been concatenated.

    Raises:
        TypeError: If the provided argument is not a list or if it contains non-string elements.
    """
    # Check that the input is a list
    if not isinstance(string_list, list):
        raise TypeError("Input must be a list")
    
    # Check that all elements in the list are strings
    for element in string_list:
        if not isinstance(element, str):
            raise TypeError("All elements in the list must be strings")
            
    return ''.join(string_list)
