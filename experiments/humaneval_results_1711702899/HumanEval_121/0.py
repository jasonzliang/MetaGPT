
def sum_odd_in_even_positions(lst):
    """Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.

    Args:
        lst (list): A list of integers.

    Returns:
        int: The sum of all odd elements at even positions in the list.

    Raises:
        ValueError: If the input is not a list or if it contains non-integer values.
    """
    
    # Checking if the input is a list and if it's not empty
    if not isinstance(lst, list) or len(lst) == 0:
        raise ValueError("Input should be a non-empty list of integers")
        
    # Checking if all elements in the list are integers
    for i in lst:
        if not isinstance(i, int):
            raise ValueError("All elements in the list should be integers")
    
    return sum(lst[i] for i in range(0, len(lst), 2) if lst[i] % 2 != 0)
