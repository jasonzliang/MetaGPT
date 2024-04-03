
def is_sorted(lst):
    """Check if a list of numbers is sorted in ascending order and has no duplicates.

    Args:
        lst: A list of integers.

    Returns:
        True if the list is sorted and has no duplicates, False otherwise.

    Raises:
        ValueError: If the input list contains negative numbers or non-integer values.
    """
    
    # Check for invalid inputs
    if not all(isinstance(i, int) and i >= 0 for i in lst):
        raise ValueError("Input list must contain only non-negative integers.")
        
    # Check if the list is sorted
    if lst != sorted(lst):
        return False
    
    # Check if there are duplicates
    if len(lst) != len(set(lst)):
        return False
    
    return True
