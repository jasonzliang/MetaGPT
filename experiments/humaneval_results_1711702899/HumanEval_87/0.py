
def get_row(lst, x):
    """Find integers x in the list and return their coordinates.

    Args:
        lst: A 2 dimensional data as a nested lists.
        x: The integer to find in the list.

    Returns:
        List of tuples (row, columns), starting with 0, sorted initially by rows in ascending order and then by columns in descending order.

    Raises:
        TypeError: If lst is not a list or x is not an integer.
    """
    if not isinstance(lst, list) or not all(isinstance(i, list) for i in lst):
        raise TypeError("lst must be a 2D list")
    if not isinstance(x, int):
        raise TypeError("x must be an integer")
    
    coordinates = []
    for row_index, row in enumerate(lst):
        for col_index in sorted([i for i, value in enumerate(row) if value == x], reverse=True):
            coordinates.append((row_index, col_index))
    
    return sorted(coordinates, key=lambda x: (x[0], -x[1]))
