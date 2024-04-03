
def get_positive(lst):
    """Return only positive numbers in the list.

    Args:
        lst (list): The input list of integers.

    Returns:
        list: A new list containing only the positive numbers from the input list.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the input list contains non-integer values.
    """
    if not isinstance(lst, list):
        raise TypeError("Input should be a list.")
    
    for i in lst:
        if not isinstance(i, int):
            raise ValueError("All elements of the input list should be integers.")
            
    return [num for num in lst if num > 0]
