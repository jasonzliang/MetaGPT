
def sort_array(array):
    """
    Given an array of non-negative integers, return a copy of the given array after sorting. 
    If the sum of the first and last elements is odd, sort in ascending order; if even, sort in descending order.
    """
    
    # Check for empty array
    if not array:
        return []
    
    # Copy the original array to avoid changing it
    sorted_array = array[:]
    
    # Calculate sum of first and last elements
    sum_first_last = array[0] + array[-1]
    
    # Sort in ascending order if sum is odd, descending if even
    if sum_first_last % 2 == 1:
        sorted_array.sort()
    else:
        sorted_array.sort(reverse=True)
        
    return sorted_array
```

# Testing the function
print(sort_array([2, 4, 3, 0, 1, 5])) # Expected output: [0, 1, 2, 3, 4, 5]
print(sort_array([2, 4, 3, 0, 1, 5, 6])) # Expected output: [6, 5, 4, 3, 2, 1, 0]
