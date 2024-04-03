
def change_base(x: int, base: int) -> str:
    """Change numerical base of input number x to base.
    Return string representation after the conversion.
    Base numbers are less than 10.
    """
    if not isinstance(x, int) or not isinstance(base, int):
        raise ValueError("Both inputs must be integers.")
    elif base < 2 or base > 9:
        raise ValueError("Base must be between 2 and 9.")
    
    digits = "0123456789"
    result = ""
    
    while x > 0:
        digit_value = x % base
        result = digits[digit_value] + result
        x //= base
        
    return result if result else '0'
