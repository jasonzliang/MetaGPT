
from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers`.
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    # Check if the input list is empty. If so, return an empty list.
    if not numbers:
        return []
    
    # Initialize a new list to store the interspersed elements.
    result = [numbers[0]]
    
    # Iterate over the rest of the elements in the input list.
    for num in numbers[1:]:
        # Append the delimeter and the current number to the result list.
        result.extend([delimeter, num])
        
    return result
