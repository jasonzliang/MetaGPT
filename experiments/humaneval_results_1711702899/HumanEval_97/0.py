
def multiply(a: int, b: int) -> int:
    """Complete the function that takes two integers and returns 
    the product of their unit digits.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the unit digits of the input integers.

    Raises:
        TypeError: If either `a` or `b` is not an integer.
    """
    
    # Check if inputs are integers
    if not all(isinstance(i, int) for i in [a, b]):
        raise TypeError("Both arguments must be integers.")
        
    return (abs(a) % 10) * (abs(b) % 10)
