
from typing import List, Tuple

def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    """ From a supplied list of numbers (of length at least two), select and return two that are the closest to each other 
        and return them in order (smaller number, larger number).
        >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
        (2.0, 2.2)
        >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
        (2.0, 2.0)
    """
    
    # Error handling: Check if the list has at least two elements
    if len(numbers) < 2:
        raise ValueError("The input list must have at least two numbers.")
        
    # Sort the list in ascending order
    sorted_numbers = sorted(numbers)
    
    # Initialize variables to keep track of the closest pair and their difference
    min_difference = float('inf')
    closest_pair = (0, 0)
    
    # Iterate over the list to find the closest pair
    for i in range(1, len(sorted_numbers)):
        diff = sorted_numbers[i] - sorted_numbers[i-1]
        
        if diff < min_difference:
            min_difference = diff
            closest_pair = (sorted_numbers[i-1], sorted_numbers[i])
    
    return closest_pair
