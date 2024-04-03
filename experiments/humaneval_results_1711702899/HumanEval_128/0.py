
def prod_signs(arr):
    """Calculate the sum of magnitudes multiplied by product of all signs in an array.

    Args:
        arr: A list of integers.

    Returns:
        The result of the operation if the input array is not empty, otherwise None.

    Raises:
        TypeError: If the input is not a list or contains non-integer elements.
    """
    # Check if input is a list
    if not isinstance(arr, list):
        raise TypeError("Input should be a list.")
    
    # Check if all elements in the list are integers
    for i in arr:
        if not isinstance(i, int):
            raise TypeError("All elements in the input list should be integers.")
    
    # Return None for empty array
    if len(arr) == 0:
        return None
    
    # Calculate product of all signs and sum of magnitudes
    prod = 1
    sum_mag = 0
    for i in arr:
        prod *= (i > 0) - (i < 0)  # Convert positive to 1, negative to -1, zero to 0
        sum_mag += abs(i)
    
    return prod * sum_mag
