
def get_positive(lst):
    """Return only positive numbers in the list.
    >>> get_positive([-1, 2, -4, 5, 6])
    [2, 5, 6]
    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    [5, 3, 2, 3, 9, 123, 1]
    """
    # Check if input is a list
    if not isinstance(lst, list):
        raise ValueError("Input must be a list")
    
    # Filter out positive numbers using list comprehension
    return [num for num in lst if num > 0]
