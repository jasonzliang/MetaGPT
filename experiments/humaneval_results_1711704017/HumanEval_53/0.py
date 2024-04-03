
def add(x: int, y: int) -> int:
    """Add two numbers x and y.

    Args:
        x (int): The first number to be added.
        y (int): The second number to be added.

    Returns:
        int: The sum of x and y.

    Raises:
        TypeError: If either x or y is not an integer.
    """
    
    # Check if inputs are integers
    if not all(isinstance(i, int) for i in [x, y]):
        raise TypeError("Both arguments must be integers.")
        
    return x + y
