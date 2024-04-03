
def generate_integers(a: int, b: int) -> list[int]:
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        A list of even numbers between a and b, sorted in ascending order.

    Raises:
        ValueError: If either a or b is not a positive integer.
    
    """
    if not (isinstance(a, int) and isinstance(b, int)):
        raise ValueError("Both inputs must be integers.")
        
    if a < 0 or b < 0:
        raise ValueError("Inputs must be positive integers.")
        
    # Swap the numbers if a > b
    if a > b:
        a, b = b, a
    
    return sorted(num for num in range(a, b + 1) if num % 2 == 0)
