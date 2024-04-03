
def common(l1: list, l2: list) -> list:
    """Return sorted unique common elements for two lists.

    Args:
        l1: The first list of integers.
        l2: The second list of integers.

    Returns:
        A sorted list of the unique common elements from both input lists.

    Raises:
        TypeError: If either of the inputs is not a list or contains non-integer values.
    """
    # Check if inputs are lists and contain only integers
    for i in [l1, l2]:
        if type(i) != list:
            raise TypeError("Both arguments must be of type 'list'.")
        elif not all(isinstance(j, int) for j in i):
            raise TypeError("All elements in the lists must be integers.")
    
    # Find common elements and return sorted unique values
    return sorted(set(l1).intersection(l2))
