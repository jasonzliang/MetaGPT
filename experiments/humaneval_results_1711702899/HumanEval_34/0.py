
def unique(lst):
    """Return sorted unique elements in a list.

    Args:
        lst (list): The input list of integers.

    Returns:
        list: A new list containing the sorted unique elements from the input list.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the input list contains non-integer values.
    """
    if not isinstance(lst, list):
        raise TypeError("Input must be a list.")
    
    for i in lst:
        if not isinstance(i, int):
            raise ValueError("All elements of the input list must be integers.")
            
    return sorted(set(lst))
