
def count_nums(arr):
    """
    This function takes an array of integers and returns the number of elements which has a sum of digits > 0. 
    If a number is negative, then its first signed digit will be negative.

    Args:
        arr: An array of integers.

    Returns:
        The count of numbers in the array whose sum of digits is greater than zero.

    Raises:
        TypeError: If the input is not a list or if any element of the list is not an integer.
    """
    
    # Checking if arr is a list
    if not isinstance(arr, list):
        raise TypeError("Input should be a list")
        
    # Checking if all elements in arr are integers
    for num in arr:
        if not isinstance(num, int):
            raise TypeError("All elements in the input list should be integers")
    
    count = 0
    for num in arr:
        # If number is negative, make it positive
        if num < 0:
            num = -num
            
        # Calculate sum of digits
        digit_sum = sum(int(digit) for digit in str(num))
        
        # Increment count if digit_sum > 0
        if digit_sum > 0:
            count += 1
    
    return count
