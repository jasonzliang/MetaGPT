
def any_int(x, y, z):
    """
    This function takes 3 numbers as input. It returns True if one of the numbers is equal to the sum of the other two, and all numbers are integers. Otherwise it returns False.

    Args:
        x (int): The first number.
        y (int): The second number.
        z (int): The third number.

    Returns:
        bool: True if one of the numbers is equal to the sum of the other two, and all numbers are integers. False otherwise.

    Raises:
        TypeError: If any of the inputs is not an integer.
    """
    
    # Checking if all inputs are integers
    if not all(isinstance(i, int) for i in [x, y, z]):
        raise TypeError("All inputs must be integers")
        
    return x == y + z or y == x + z or z == x + y
