
def x_or_y(n, x, y):
    """A simple program which returns the value of x if n is a prime number 
    and returns the value of y otherwise.

    Args:
        n: The number to check for primality.
        x: The first value to return.
        y: The second value to return.

    Returns:
        If n is a prime number, returns x. Otherwise, returns y.

    Raises:
        TypeError: If n, x or y are not integers.
        ValueError: If n is less than 2 (as there are no prime numbers less than 2).
    """
    
    # Check if inputs are valid
    if not all(isinstance(i, int) for i in [n, x, y]):
        raise TypeError("All arguments must be integers.")
    if n < 2:
        raise ValueError("There are no prime numbers less than 2.")
    
    # Check if n is a prime number
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return y
    return x
