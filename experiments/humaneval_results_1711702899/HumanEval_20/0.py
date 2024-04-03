
from typing import List, Tuple

def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    """From a supplied list of numbers (of length at least two), select and return two that are the closest to each other.
    
    Args:
        numbers: A list of floats with at least two elements.
        
    Returns:
        A tuple containing the two closest numbers from the input list, in ascending order.
        
    Raises:
        ValueError: If the input list has less than two elements.
     """
    if len(numbers) < 2:
        raise ValueError("The input list must have at least two elements.")
    
    numbers.sort()  # Sorting the list in ascending order
    closest_pair = (numbers[0], numbers[1])  # Initialize with first two elements
    min_difference = abs(closest_pair[0] - closest_pair[1])  # Initial minimum difference
    
    for i in range(1, len(numbers) - 1):
        current_pair = (numbers[i], numbers[i + 1])
        diff = abs(current_pair[0] - current_pair[1])
        
        if diff < min_difference:
            closest_pair = current_pair
            min_difference = diff
            
    return closest_pair if closest_pair[0] <= closest_pair[1] else (closest_pair[1], closest_pair[0])
