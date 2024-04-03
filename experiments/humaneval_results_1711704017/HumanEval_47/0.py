
def median(lst):
    """Return the median of elements in the list lst.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """
    # Error handling: check if the list is empty
    if not lst:
        return "Error: The list is empty."

    # Sort the list in ascending order
    sorted_lst = sorted(lst)

    # Calculate the median
    n = len(sorted_lst)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
    else:
        return sorted_lst[mid]
