
def can_arrange(arr):
    """Create a function which returns the largest index of an element which
    is not greater than or equal to the element immediately preceding it. If
    no such element exists then return -1. The given array will not contain
    duplicate values.

    Examples:
    can_arrange([1,2,4,3,5]) = 3
    can_arrange([1,2,3]) = -1
    
    Args:
        arr: A list of integers in non-decreasing order.

    Returns:
        The largest index of an element which is not greater than or equal to the 
        element immediately preceding it. If no such element exists then return -1.

    Raises:
        TypeError: If arr is not a list.
        ValueError: If arr contains non-integer values.
    """
    
    # Check if arr is a list
    if not isinstance(arr, list):
        raise TypeError("Input should be a list.")
        
    # Check if all elements in arr are integers
    for i in arr:
        if not isinstance(i, int):
            raise ValueError("All elements in the input list should be integers.")
    
    # Iterate over the array from right to left
    for i in range(len(arr)-1, 0, -1):
        # If current element is less than previous one return its index
        if arr[i] < arr[i-1]:
            return i
    
    # If no such element exists then return -1
    return -1
