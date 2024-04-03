
from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers`

    Args:
        numbers: A list of integers.
        delimeter: An integer to be inserted between each pair of consecutive elements in the list.

    Returns:
        A new list with the 'delimeter' inserted between each pair of consecutive elements from the input list.

    Raises:
        TypeError: If `numbers` is not a list or if any element of `numbers` is not an integer, or if `delimeter` is not an integer.
        
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not isinstance(numbers, list):
        raise TypeError("'numbers' must be a list")
    
    if not all(isinstance(i, int) for i in numbers):
        raise TypeError("All elements of 'numbers' must be integers")
        
    if not isinstance(delimeter, int):
        raise TypeError("'delimeter' must be an integer")
    
    result = []
    for num in numbers:
        result.append(num)
        result.append(delimeter)
    
    # Remove the last 'delimeter' as it is after the last number
    if len(result) > 0:
        result.pop()
        
    return result
