
def choose_num(x: int, y: int) -> int:
    """This function takes two positive numbers x and y and returns the
    biggest even integer number that is in the range [x, y] inclusive. If 
    there's no such number, then the function should return -1.

    Args:
        x: The first positive number.
        y: The second positive number.

    Returns:
        The biggest even integer number in the range [x, y] inclusive or -1 if 
        there's no such number.

    Raises:
        ValueError: If either x or y is not a positive integer.
    """
    # Check that both inputs are positive integers
    if not (isinstance(x, int) and isinstance(y, int)) or x <= 0 or y <= 0:
        raise ValueError("Both x and y must be positive integers.")
    
    # If y is even, return it
    if y % 2 == 0:
        return y
    
    # Otherwise, if x is greater than y, there's no number in the range
    elif x > y:
        return -1
    
    # Otherwise, the biggest even number is one less than y
    else:
        return y - 1
