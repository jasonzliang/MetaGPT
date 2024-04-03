
def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.
    """
    # Error handling for invalid inputs
    if not isinstance(arr, list) or not all(isinstance(i, int) for i in arr):
        raise ValueError("The array should only contain integers.")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("K should be a non-negative integer.")
    elif len(arr) < k:
        raise ValueError("Length of the array should be greater than or equal to K.")
    
    # Sorting the array in descending order and returning first k elements
    arr.sort(reverse=True)
    return arr[:k]
