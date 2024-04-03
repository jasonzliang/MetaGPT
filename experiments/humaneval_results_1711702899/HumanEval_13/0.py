
def greatest_common_divisor(a: int, b: int) -> int:
    """Return the greatest common divisor of two integers a and b.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The greatest common divisor of a and b.

    Raises:
        ValueError: If either a or b is not an integer.
        ZeroDivisionError: If both a and b are zero.
    """
    if not all(isinstance(i, int) for i in (a, b)):
        raise ValueError("Both inputs must be integers.")
    
    if a == 0 and b == 0:
        raise ZeroDivisionError("At least one of the inputs should be non-zero.")
        
    while b != 0:
        a, b = b, a % b
    return abs(a)
