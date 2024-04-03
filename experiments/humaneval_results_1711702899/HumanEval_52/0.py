
def below_threshold(lst: list, threshold: int) -> bool:
    """Return True if all numbers in the list l are below threshold t.

    Args:
        lst: A list of integers.
        threshold: An integer that serves as a limit for comparison with the elements in the list.

    Returns:
        A boolean value indicating whether all numbers in the list are less than the threshold.

    Raises:
        TypeError: If the input list contains non-integer values or if the threshold is not an integer.
    """
    
    # Checking if all elements in the list are integers
    if not all(isinstance(i, int) for i in lst):
        raise TypeError("All elements in the list must be integers.")
        
    # Checking if threshold is an integer
    if not isinstance(threshold, int):
        raise TypeError("Threshold must be an integer.")
    
    return all(i < threshold for i in lst)
