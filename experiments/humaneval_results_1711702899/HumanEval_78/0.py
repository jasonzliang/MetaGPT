
def hex_key(num):
    """Counts the number of hexadecimal digits in a string that are primes.

    Args:
        num: A string representing a hexadecimal number.

    Returns:
        The count of prime hexadecimal digits in the input string.

    Raises:
        TypeError: If the input is not a string.
    """
    
    if not isinstance(num, str):
        raise TypeError('Input must be a string')
        
    primes = ['2', '3', '5', '7', 'B', 'D']  # hexadecimal digits that are prime
    count = 0
    
    for digit in num:
        if digit.upper() in primes:
            count += 1
            
    return count
