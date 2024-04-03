
def prod_signs(arr):
    """
    Calculate the sum of magnitudes of integers multiplied by product of all signs.
    Return None for empty arr.
    """
    if not arr:  # Check if array is empty
        return None
    
    sign = 1
    magnitude_sum = 0
    
    for num in arr:
        if num < 0:
            sign *= -1
        elif num == 0:
            return 0
        
        magnitude_sum += abs(num)
    
    return sign * magnitude_sum
