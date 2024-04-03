
def total_match(lst1, lst2):
    """
    This function accepts two lists of strings and returns the list that has 
    total number of chars in the all strings of the list less than the other list.
    If the two lists have the same number of chars, return the first list.

    Args:
        lst1: The first list of strings.
        lst2: The second list of strings.

    Returns:
        The list that has total number of chars in the all strings of the list less than 
        the other list or the first list if both lists have the same number of chars.

    Raises:
        TypeError: If either lst1 or lst2 is not a list.
        ValueError: If either lst1 or lst2 contains non-string elements.
    """
    
    # Check if both inputs are lists
    if not (isinstance(lst1, list) and isinstance(lst2, list)):
        raise TypeError("Both arguments must be of type list.")
        
    # Check if both lists contain only strings
    for i in lst1 + lst2:
        if not isinstance(i, str):
            raise ValueError("All elements in the lists must be of type string.")
    
    # Calculate total number of chars in each list
    total_chars_lst1 = sum(len(s) for s in lst1)
    total_chars_lst2 = sum(len(s) for s in lst2)
    
    # Return the list with less total number of chars or the first one if equal
    return lst1 if total_chars_lst1 <= total_chars_lst2 else lst2
