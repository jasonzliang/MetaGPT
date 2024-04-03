
def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.

    Args:
        arr: A list of integers.
        k: An integer representing the number of maximum elements to be returned.

    Returns:
        A sorted list of the k maximum elements from arr.

    Raises:
        ValueError: If k is greater than the length of arr or less than 0.
    
    """
    if not (0 <= k <= len(arr)):
        raise ValueError("k must be between 0 and the length of arr")
        
    # Sorting the array in descending order and returning the first k elements
    return sorted(arr, reverse=True)[:k]
