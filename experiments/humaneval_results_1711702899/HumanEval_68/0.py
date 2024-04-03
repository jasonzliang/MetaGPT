
def pluck(arr):
    """
    Given an array representing a branch of a tree that has non-negative integer nodes
    this function will pluck one of the nodes with smallest even value and return it.
    If multiple nodes with the same smallest even value are found, it returns the node that has smallest index.
    The plucked node is returned in a list [smallest_value, its index].
    If there are no even values or the given array is empty, it returns an empty list.

    Args:
        arr (List[int]): An array representing a branch of a tree with non-negative integer nodes.

    Returns:
        List[Union[int, str], Union[int, str]]: A list containing the smallest even value and its index. 
                                              If there are no even values or the given array is empty, it returns an empty list.
    
    Raises:
        ValueError: If the input array contains non-integer elements.
    """
    # Check if all elements in arr are integers
    for i in arr:
        if not isinstance(i, int):
            raise ValueError("All elements in the array should be integers.")
    
    # Filter out even numbers and their indices
    evens = [(val, idx) for idx, val in enumerate(arr) if val % 2 == 0]
    
    # If there are no even numbers return an empty list
    if not evens:
        return []
    
    # Sort the list of tuples by value and index
    evens.sort()
    
    # Return the smallest even number and its index
    return [evens[0][0], evens[0][1]]
