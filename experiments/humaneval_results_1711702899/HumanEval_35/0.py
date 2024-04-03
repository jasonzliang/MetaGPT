
def max_element(lst: list):
    """Return maximum element in the list.

    Args:
        lst (list): The input list of numbers.

    Returns:
        int or float: The maximum number in the list. If the list is empty, returns None.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list does not contain any numeric elements.
    """
    if not isinstance(lst, list):
        raise TypeError("Input should be a list.")
    
    if len(lst) == 0:
        return None
    
    try:
        max_value = max(lst)
    except TypeError as e:
        raise ValueError("List should contain only numeric elements.") from e
        
    return max_value
