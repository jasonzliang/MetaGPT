
def sort_third(lst):
    """This function takes a list lst and returns a new list where the elements at indices divisible by three are sorted.

    Args:
        lst (list): The input list to be processed.

    Returns:
        list: A new list with elements at indices divisible by three sorted.

    Raises:
        TypeError: If the input is not a list.
    """
    
    # Check if the input is a list
    if not isinstance(lst, list):
        raise TypeError("Input should be a list")
        
    # Create a copy of the original list
    sorted_list = lst[:]
    
    # Get elements at indices divisible by three and sort them
    thirds = sorted([val for i, val in enumerate(lst) if i % 3 == 0])
    
    # Replace elements at indices divisible by three with the sorted ones
    for i in range(len(sorted_list)):
        if i % 3 == 0:
            sorted_list[i] = thirds.pop(0)
            
    return sorted_list
