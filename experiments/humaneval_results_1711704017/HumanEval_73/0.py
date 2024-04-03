
def smallest_change(arr):
    """
    Given an array arr of integers, find the minimum number of elements that
    need to be changed to make the array palindromic. A palindromic array is an array that
    is read the same backwards and forwards. In one change, you can change one element to any other element.

    For example:
    smallest_change([1,2,3,5,4,7,9,6]) == 4
    smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1
    smallest_change([1, 2, 3, 2, 1]) == 0
    """
    
    # Initialize a counter for changes and two pointers at the start and end of the array.
    changes = 0
    left = 0
    right = len(arr) - 1

    # Iterate over the array from both ends towards the center.
    while left < right:
        # If elements at opposite ends are not equal, increment the counter and move pointers.
        if arr[left] != arr[right]:
            changes += 1
        
        left += 1
        right -= 1
    
    return changes
