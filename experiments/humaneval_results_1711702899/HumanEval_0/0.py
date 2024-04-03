
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.

    Args:
        numbers: A list of floating point numbers.
        threshold: The maximum difference between two numbers for them to be considered 'close'.

    Returns:
        True if there are two numbers in the list that are closer to each other than the 
        given threshold, False otherwise.

    Raises:
        ValueError: If the input list is empty or if the threshold is not a positive number.
    """
    
    # Check for invalid inputs
    if len(numbers) == 0:
        raise ValueError("The input list cannot be empty.")
    if threshold <= 0:
        raise ValueError("Threshold must be a positive number.")
        
    # Sort the numbers in ascending order
    sorted_numbers = sorted(numbers)
    
    # Check for close elements
    for i in range(len(sorted_numbers)-1):
        if abs(sorted_numbers[i] - sorted_numbers[i+1]) <= threshold:
            return True
            
    return False
