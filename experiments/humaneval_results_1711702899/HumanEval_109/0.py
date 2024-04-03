
def move_one_ball(arr):
    """Determine if it is possible to get an array sorted in non-decreasing order by performing right shift operation.

    Args:
        arr: A list of integers.

    Returns:
        True if the given array can be sorted in non-decreasing order by performing at most one right shift operation, False otherwise.

    Raises:
        ValueError: If the input is not a list or if it contains non-integer elements.
    """
    
    # Checking if arr is a list and containing only integers
    if not isinstance(arr, list):
        raise ValueError("Input should be a list")
    for i in arr:
        if not isinstance(i, int):
            raise ValueError("List should contain only integers")
    
    # If the array is empty or contains one element it's already sorted
    if len(arr) < 2:
        return True
    
    # Finding the index of the maximum value in arr
    max_index = arr.index(max(arr))
    
    # Checking if the array can be sorted by performing at most one right shift operation
    for i in range(len(arr) - 1):
        if arr[(i + max_index + 1) % len(arr)] < arr[(i + max_index) % len(arr)]:
            return False
    
    return True
