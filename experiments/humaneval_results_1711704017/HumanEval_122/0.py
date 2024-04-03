
def add_elements(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return
    the sum of the elements with at most two digits from the first k elements of arr.
    """
    # Error handling for invalid inputs
    if not isinstance(arr, list) or not all(isinstance(i, int) for i in arr):
        raise ValueError("The array should only contain integers.")
    if not isinstance(k, int) or k <= 0 or k > len(arr):
        raise ValueError("Invalid value of k. It should be an integer between 1 and the length of the array.")
    
    # Initialize sum to zero
    total_sum = 0
    
    # Iterate over the first k elements in arr
    for i in range(k):
        # If the element has at most two digits, add it to the sum
        if abs(arr[i]) <= 99:
            total_sum += arr[i]
            
    return total_sum
