
from typing import List, Any

def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers.

    Args:
        values: A list of any python values.

    Returns:
        A list of integers from the input list.

    Raises:
        TypeError: If the input is not a list.
    """
    if not isinstance(values, list):
        raise TypeError('The input must be a list')
    
    return [value for value in values if isinstance(value, int)]
