
def add_elements(arr, k):
    """Given a non-empty array of integers arr and an integer k, 
       return the sum of the elements with at most two digits from the first k elements of arr.

    Args:
        arr: A list of integers.
        k: An integer indicating how many elements to consider from the start of arr.

    Returns:
        The sum of the elements in arr with at most two digits, considering only the first k elements.

    Raises:
        ValueError: If k is not within the valid range (1 <= k <= len(arr)).
        TypeError: If arr contains non-integer values or if k is not an integer.
    """
    
    # Checking if inputs are valid
    if not all(isinstance(i, int) for i in arr):
        raise TypeError("All elements of the array must be integers.")
    if not isinstance(k, int):
        raise TypeError("The number of elements to consider must be an integer.")
    if k < 1 or k > len(arr):
        raise ValueError("The number of elements to consider must be within 1 and the length of the array.")
    
    # Calculating sum of elements with at most two digits
    return sum(i for i in arr[:k] if i <= 99)
