
def strange_sort_list(lst):
    """Given a list of integers, returns the list sorted in a 'strange' order. 
    Strange sorting is when you start with the minimum value, then maximum of the remaining integers, 
    then minimum and so on.

    Args:
        lst (list): A list of integers.

    Returns:
        list: The sorted list in 'strange' order.

    Raises:
        TypeError: If the input is not a list or if it contains non-integer elements.
    """
    
    # Checking if lst is a list and all its elements are integers
    if not isinstance(lst, list):
        raise TypeError("Input should be a list.")
    for i in lst:
        if not isinstance(i, int):
            raise TypeError("All elements of the input list should be integers.")
    
    # Sorting the list in ascending order
    sorted_lst = sorted(lst)
    
    result = []
    while len(sorted_lst) > 0:
        # Appending minimum element
        result.append(sorted_lst[0])
        sorted_lst.pop(0)
        
        if len(sorted_lst) > 0:
            # Appending maximum element
            result.append(sorted_lst[-1])
            sorted_lst.pop(-1)
    
    return result
