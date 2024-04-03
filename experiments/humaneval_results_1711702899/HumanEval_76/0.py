
def is_simple_power(x: int, n: int) -> bool:
    """Checks if a number x is a simple power of n.

    Args:
        x: The number to check.
        n: The base number.

    Returns:
        True if x is a simple power of n, False otherwise.

    Raises:
        ValueError: If x or n are less than 1.
    """
    if x < 1 or n < 1:
        raise ValueError("Both x and n must be greater than 0.")
    
    while x % n == 0:
        x /= n
        
    return x == 1
