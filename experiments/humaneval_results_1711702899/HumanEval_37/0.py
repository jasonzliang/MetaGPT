
def sort_even(lst):
    """This function takes a list lst and returns a new list where the elements at even indices are sorted.

    Args:
        lst (list): The input list to be processed.

    Returns:
        list: A new list with elements at even indices sorted.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list contains non-integer values.
    """
    
    # Check if lst is a list
    if not isinstance(lst, list):
        raise TypeError("Input should be a list.")
        
    # Check if all elements in lst are integers
    for i in lst:
        if not isinstance(i, int):
            raise ValueError("All elements in the list should be integers.")
    
    # Create a new list with elements at even indices sorted
    sorted_even = [x for x in sorted(lst[::2])]
    lst[::2] = sorted_even
    
    return lst
