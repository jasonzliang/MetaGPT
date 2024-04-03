
def circular_shift(x, shift):
    """Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(123456, 2)
    "561234"
    >>> circular_shift(123456, 7)
    "654321"
    """
    # Convert the integer to a string for easy manipulation
    x = str(x)

    # If shift is greater than number of digits, reverse the string
    if shift > len(x):
        return x[::-1]
    
    # Circular shift right by shift places
    else:
        return x[-shift:] + x[:-shift]
