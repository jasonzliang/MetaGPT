
def change_base(x: int, base: int) -> str:
    """Change numerical base of input number x to base.

    Args:
        x (int): The number to be converted.
        base (int): The base to convert the number into.

    Returns:
        str: The string representation after the conversion.

    Raises:
        ValueError: If base is greater than 10 or less than 2.

    """
    
    if not(2 <= base <= 10):
        raise ValueError("Base must be between 2 and 10")
        
    digits = "0123456789"
    result = ""
    
    while x > 0:
        digit_value = x % base
        result = digits[digit_value] + result
        x //= base
        
    return result if result else "0"
