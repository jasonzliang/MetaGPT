
def circular_shift(x: int, shift: int) -> str:
    """Circular shift the digits of the integer x, shift the digits right by shift 
    and return the result as a string. If shift > number of digits, return digits reversed.

    Args:
        x (int): The integer to be shifted.
        shift (int): The amount to shift x.

    Returns:
        str: The circularly shifted integer as a string.

    Raises:
        TypeError: If x or shift is not an integer.
        ValueError: If x or shift is negative.
    """
    # Check if inputs are integers
    if not all(isinstance(i, int) for i in [x, shift]):
        raise TypeError("Both x and shift must be integers.")
    
    # Check if inputs are non-negative
    if any(i < 0 for i in [x, shift]):
        raise ValueError("Both x and shift must be non-negative.")
        
    str_x = str(x)  # Convert integer to string
    len_str_x = len(str_x)  # Get the length of the string
    
    if shift > len_str_x:  # If shift is greater than number of digits, reverse the digits
        return str_x[::-1]
    else:  # Otherwise, perform circular shift
        return str_x[-shift:] + str_x[:-shift]
