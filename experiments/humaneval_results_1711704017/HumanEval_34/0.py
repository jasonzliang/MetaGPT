
def unique(lst):
    """Return sorted unique elements in a list.
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
    # Error handling: Check if input is a list.
    if not isinstance(lst, list):
        raise ValueError("Input must be a list.")
    
    # Use set to remove duplicates and sorted function to sort the result.
    return sorted(set(lst))
