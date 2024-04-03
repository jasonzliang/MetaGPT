
def next_smallest(lst):
    """
    This function returns the 2nd smallest element of the list.

    Args:
        lst: A list of integers.

    Returns:
        The second smallest integer in the list, or None if there is no such element.

    Raises:
        ValueError: If the input is not a list or if it contains non-integer elements.
    """
    
    # Check that lst is a list and contains only integers
    if not isinstance(lst, list):
        raise ValueError("Input must be a list.")
    for i in lst:
        if not isinstance(i, int):
            raise ValueError("List must contain only integers.")
    
    # Remove duplicates and sort the list
    unique_sorted = sorted(set(lst))
    
    # If there are at least two elements in the list, return the second smallest
    if len(unique_sorted) > 1:
        return unique_sorted[1]
    else:
        return None
