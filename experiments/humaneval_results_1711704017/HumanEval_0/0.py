
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Sort the list of numbers in ascending order
    sorted_numbers = sorted(numbers)
    
    # Iterate over each pair of adjacent numbers in the sorted list
    for i in range(len(sorted_numbers)-1):
        if abs(sorted_numbers[i] - sorted_numbers[i+1]) < threshold:
            return True  # If the difference is less than the threshold, return True
    
    return False  # If no such pair found, return False
