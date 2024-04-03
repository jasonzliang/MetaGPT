
def next_smallest(lst):
    """
    Returns the second smallest element in a list.
    If there is no such element or the input is not valid, returns None.
    """
    # Error handling for invalid inputs
    if not isinstance(lst, list) or len(lst) < 2:
        return None
    
    # Remove duplicates and sort the list
    unique_sorted = sorted(set(lst))
    
    # If there are at least two distinct elements in the list, return the second smallest
    if len(unique_sorted) >= 2:
        return unique_sorted[1]
    else:
        return None
