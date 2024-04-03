
def incr_list(lst):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [6, 4, 6, 3, 4, 4, 10, 1, 124]
    """
    # Error handling: check if input is a list
    if not isinstance(lst, list):
        raise ValueError("Input must be a list")
    
    # Use list comprehension to increment each element by 1
    return [i + 1 for i in lst]
