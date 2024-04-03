
def pairs_sum_to_zero(l):
    """
    This function takes a list of integers as an input. 
    It returns True if there are two distinct elements in the list that sum to zero, and False otherwise.

    Args:
        l (list): A list of integers.

    Returns:
        bool: True if there are two distinct elements in the list that sum to zero, False otherwise.

    Raises:
        TypeError: If the input is not a list or if the list contains non-integer values.
    """
    
    # Checking if the input is a list and if it contains only integers
    if not isinstance(l, list):
        raise TypeError("The input should be a list.")
    for i in l:
        if not isinstance(i, int):
            raise TypeError("All elements of the list should be integers.")
    
    # Checking if there are two distinct elements that sum to zero
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] + l[j] == 0:
                return True
    return False
