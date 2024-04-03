
from typing import List

def rolling_max(numbers: List[int]) -> List[int]:
    """From a given list of integers, generate a list of rolling maximum element found until given moment in the sequence.

    Args:
        numbers: A list of integers.

    Returns:
        A list of rolling maximum elements.

    Raises:
        TypeError: If input is not a list or if it contains non-integer values.

    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
    # Check if input is a list of integers
    if not isinstance(numbers, list) or not all(isinstance(i, int) for i in numbers):
        raise TypeError("Input must be a list of integers.")
    
    max_list = []
    current_max = None
    for num in numbers:
        if current_max is None or num > current_max:
            current_max = num
        max_list.append(current_max)
        
    return max_list
