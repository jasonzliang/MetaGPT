
def greatest_common_divisor(a: int, b: int) -> int:
    """Return the greatest common divisor of two integers a and b.

    This function uses the Euclidean algorithm to find the GCD. The algorithm works by repeatedly 
    replacing (a, b) with (b, a mod b) until b is zero. At this point, a is the GCD.
    
    Args:
        a: An integer.
        b: Another integer.
        
    Returns:
        The greatest common divisor of a and b.
    """
    while b != 0:
        a, b = b, a % b
    return abs(a)
