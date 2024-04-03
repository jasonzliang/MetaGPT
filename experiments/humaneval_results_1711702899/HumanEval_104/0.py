
def unique_digits(numbers):
    """Given a list of positive integers numbers, returns a sorted list of all 
    elements that has no even digit.

    Args:
        numbers: A list of positive integers.

    Returns:
        A sorted list of all elements in the input list that has no even digits.

    Raises:
        ValueError: If any element in the input list is not a positive integer.
    """
    
    # Check if all numbers are positive integers
    for num in numbers:
        if not isinstance(num, int) or num <= 0:
            raise ValueError("All elements in the list must be positive integers.")
            
    # Filter out numbers with even digits and sort the result
    return sorted([num for num in numbers if all(digit % 2 == 1 for digit in map(int, str(num)))])
