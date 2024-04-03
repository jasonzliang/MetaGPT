
def move_one_ball(arr):
    """Check if it is possible to sort the array by performing right shift operation."""
    # If the array is empty, return True.
    if not arr:
        return True

    # Find the index of the maximum element in the array.
    max_index = arr.index(max(arr))
    
    # Check if the maximum element is at the end of the array. If it is, then we cannot sort the array by performing right shift operation.
    if max_index == len(arr) - 1:
        return False
    else:
        return True
