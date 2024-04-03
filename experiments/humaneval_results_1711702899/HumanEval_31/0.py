
def is_prime(n):
    """Return true if a given number is prime, and false otherwise.

    Args:
        n (int): The number to check for primality.

    Returns:
        bool: True if the number is prime, False otherwise.

    Raises:
        TypeError: If the input is not an integer.
        ValueError: If the input is less than 2.
    """
    # Check that n is a positive integer
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    elif n < 2:
        raise ValueError("Input must be greater than or equal to 2.")
    
    # Check for factors up to sqrt(n)
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
            
    return True
