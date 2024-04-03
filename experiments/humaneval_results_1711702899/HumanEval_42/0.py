
def incr_list(lst: list) -> list:
    """Return a new list where each element is incremented by 1.

    Args:
        lst: The input list of integers.

    Returns:
        A new list with elements incremented by 1.

    Raises:
        TypeError: If the provided argument is not a list.
        ValueError: If the provided list contains non-integer values.
    """
    if not isinstance(lst, list):
        raise TypeError("The input should be a list.")
    
    try:
        return [x + 1 for x in lst]
    except TypeError as e:
        raise ValueError("All elements of the list should be integers.") from e
