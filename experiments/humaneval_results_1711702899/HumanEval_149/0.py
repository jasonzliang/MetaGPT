
def sorted_list_sum(lst):
    """Deletes the strings that have odd lengths from a list of strings, 
    sorts the resulted list by length in ascending order, and returns it.

    Args:
        lst (List[str]): A list of strings.

    Returns:
        List[str]: The sorted list with even-length strings only.

    Raises:
        TypeError: If the input is not a list or if any element in the list is not a string.
    """
    
    # Checking if lst is a list and all elements are strings
    if not isinstance(lst, list):
        raise TypeError("Input should be a list.")
    for i in lst:
        if not isinstance(i, str):
            raise TypeError("All elements of the input list should be strings.")
    
    # Filtering out odd-length strings and sorting by length and alphabetically
    result = sorted([s for s in lst if len(s) % 2 == 0], key=lambda x: (len(x), x))
    
    return result
