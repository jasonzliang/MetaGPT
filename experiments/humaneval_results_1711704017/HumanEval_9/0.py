
from typing import List

def rolling_max(numbers: List[int]) -> List[int]:
    """ From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
    if not numbers:
        return []

    max_num = numbers[0]
    result = []
    
    for num in numbers:
        # Update the maximum number if current number is greater than max_num
        if num > max_num:
            max_num = num
        
        result.append(max_num)
    
    return result
