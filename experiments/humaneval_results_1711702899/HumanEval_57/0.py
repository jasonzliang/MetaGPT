
def monotonic(lst):
    """Return True if list elements are monotonically increasing or decreasing.

    Args:
        lst (list): The input list to check for monotonicity.

    Returns:
        bool: True if the list is monotonic, False otherwise.

    Raises:
        TypeError: If the input is not a list.
    """
    # Check that the input is a list
    if not isinstance(lst, list):
        raise TypeError("Input must be a list")
    
    # Return True for empty or single-element lists
    if len(lst) < 2:
        return True
    
    # Check if the list is monotonic increasing
    diffs = [x - i for i, x in enumerate(lst)]
    if all(diff > 0 for diff in diffs):
        return True
    
    # Check if the list is monotonic decreasing
    if all(diff < 0 for diff in diffs):
        return True
    
    # If neither condition is met, the list is not monotonic
    return False
