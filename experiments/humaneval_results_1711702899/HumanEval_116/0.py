
def sort_array(arr):
    """
    This function sorts an array of non-negative integers according to the number of ones in their binary representation 
    in ascending order. For similar number of ones, it sorts based on decimal value.

    Args:
        arr: A list of non-negative integers.

    Returns:
        The sorted list of non-negative integers.

    Raises:
        ValueError: If the input list contains negative numbers.
        
    """
    
    # Check if all elements in the array are non-negative
    for num in arr:
        if num < 0:
            raise ValueError("All elements in the array must be non-negative.")
            
    # Sort the array based on the number of ones in their binary representation and decimal value
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))
