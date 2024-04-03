
def smallest_change(arr):
    """
    Given an array arr of integers, find the minimum number of elements that need to be changed 
    to make the array palindromic. A palindromic array is an array that is read the same backwards and forwards. 
    In one change, you can change one element to any other element.

    Args:
        arr: The input list of integers.

    Returns:
        The minimum number of elements that need to be changed to make the array palindromic.

    Raises:
        TypeError: If the input is not a list or if it contains non-integer values.
    """
    
    # Checking if arr is a list and if all its elements are integers
    if not isinstance(arr, list):
        raise TypeError("The input should be a list.")
    for i in arr:
        if not isinstance(i, int):
            raise TypeError("All elements of the list should be integers.")
    
    # Initializing variables
    n = len(arr)
    changes = 0
    
    # Comparing corresponding elements from start and end of array
    for i in range(n // 2):
        if arr[i] != arr[n - i - 1]:
            changes += 1
            
    return changes
