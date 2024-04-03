
def max_element(lst):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
    # Error handling: check if the input is a list
    if not isinstance(lst, list):
        return "Error: Input should be a list."
        
    # Error handling: check if the list is empty
    if len(lst) == 0:
        return "Error: The list is empty."
    
    max_value = lst[0]  # Initialize maximum value as first element of list
    
    for i in range(1, len(lst)):  # Iterate over the rest of the list
        if lst[i] > max_value:  # If current element is greater than max_value
            max_value = lst[i]  # Update max_value
    
    return max_value
