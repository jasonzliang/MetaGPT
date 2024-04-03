
def special_factorial(n):
    """Calculates the Brazilian factorial of a given number.

    Args:
        n (int): The number to calculate the Brazilian factorial for.

    Returns:
        int: The result of the Brazilian factorial operation.

    Raises:
        ValueError: If the input is not an integer or if it's less than 0.
    """
    # Checking if n is a non-negative integer
    if not isinstance(n, int) or n < 0:
        raise ValueError("The input must be a non-negative integer.")
    
    # Calculating the factorial
    result = 1
    for i in range(1, n+1):
        for j in range(1, i+1):
            result *= j
            
    return result
