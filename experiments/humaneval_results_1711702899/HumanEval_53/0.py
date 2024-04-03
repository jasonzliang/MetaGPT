
def add(x: int, y: int) -> int:
    """Add two numbers x and y.

    Args:
        x: The first number to be added.
        y: The second number to be added.

    Returns:
        The sum of the two input numbers.

    Raises:
        TypeError: If either x or y is not an integer.
    """
    if not all(isinstance(i, int) for i in (x, y)):
        raise TypeError('Both arguments must be integers')
    
    return x + y
