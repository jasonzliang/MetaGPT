
from typing import List

def rescale_to_unit(numbers: List[float]) -> List[float]:
    """Given a list of numbers (of at least two elements), apply a linear transform to that list, 
    such that the smallest number will become 0 and the largest will become 1.

    Args:
        numbers: A list of floats with at least two elements.

    Returns:
        A new list where each element is rescaled based on its position in the sorted version of the input list.

    Raises:
        ValueError: If the input list has less than two elements or if any element is not a number.
    """
    
    # Checking for valid input
    if len(numbers) < 2:
        raise ValueError("The input list must have at least two elements.")
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements of the input list must be numbers.")
    
    # Sorting the numbers and finding min and max
    sorted_numbers = sorted(numbers)
    smallest = sorted_numbers[0]
    largest = sorted_numbers[-1]
    
    # Calculating range
    rng = largest - smallest if largest != smallest else 1
    
    # Rescaling each number and returning the result
    return [(x - smallest) / rng for x in numbers]
