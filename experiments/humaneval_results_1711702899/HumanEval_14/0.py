
from typing import List

def all_prefixes(input_string: str) -> List[str]:
    """Return list of all prefixes from shortest to longest of the input string.

    Args:
        input_string: The string for which we want to find all its prefixes.

    Returns:
        A list of all prefixes of the input string, sorted by length in ascending order.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(input_string, str):
        raise TypeError('The input must be a string')
    
    return [input_string[:i] for i in range(1, len(input_string) + 1)]
