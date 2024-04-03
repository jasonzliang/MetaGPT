
def median(lst):
    """Return the median of elements in the list lst.

    Args:
        lst: A list of numbers.

    Returns:
        The median value of the list. If the length of the list is even, 
        the average of the two middle values will be returned.

    Raises:
        TypeError: If the input is not a list or if it contains non-numeric elements.
        ValueError: If the input list is empty.
    """
    # Check that lst is a list and is not empty
    if not isinstance(lst, list) or len(lst) == 0:
        raise TypeError("Input must be a non-empty list")
    
    # Check that all elements in the list are numbers
    for i in lst:
        if not isinstance(i, (int, float)):
            raise TypeError("All elements in the list must be numeric")
            
    # Sort the list
    lst.sort()
    
    # Calculate the median
    n = len(lst)
    mid = n // 2
    if n % 2 == 0:
        return (lst[mid - 1] + lst[mid]) / 2
    else:
        return lst[mid]
