
from typing import List, Tuple

def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """For a given list of integers, return a tuple consisting of a sum 
       and a product of all the integers in a list. Empty sum should be equal to 0 
       and empty product should be equal to 1.

    Args:
        numbers: A list of integers.

    Returns:
        A tuple where the first element is the sum of all elements in the input list, 
        and the second element is the product of all elements in the input list.

    Raises:
        TypeError: If the input is not a list or if it contains non-integer values.
    """
    # Check that the input is a list of integers
    if not isinstance(numbers, list) or not all(isinstance(i, int) for i in numbers):
        raise TypeError("Input must be a list of integers.")
    
    # Calculate sum and product
    total_sum = sum(numbers)
    total_product = 1
    for num in numbers:
        total_product *= num
        
    return (total_sum, total_product)
