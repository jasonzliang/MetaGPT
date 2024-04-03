
def sort_array(arr):
    """
    Sort an array of non-negative integers according to number of ones in their binary representation.
    For similar number of ones, sort based on decimal value.
    """
    # Check if all elements are non-negative integers
    for i in arr:
        if not isinstance(i, int) or i < 0:
            raise ValueError("All elements must be non-negative integers")
    
    # Sort the array based on number of ones and decimal value
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))
