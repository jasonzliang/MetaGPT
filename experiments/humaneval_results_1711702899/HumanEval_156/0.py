
def int_to_mini_roman(number):
    """
    Given a positive integer, obtain its roman numeral equivalent as a string,
    and return it in lowercase.

    Args:
        number (int): A positive integer between 1 and 1000 inclusive.

    Returns:
        str: The roman numeral equivalent of the input number in lowercase.

    Raises:
        ValueError: If the input number is not a positive integer or if it's outside the range 1 to 1000.
        
    Examples:
    >>> int_to_mini_roman(19)
    'xix'
    >>> int_to_mini_roman(152)
    'clii'
    >>> int_to_mini_roman(426)
    'cdxxvi'
    """
    if not isinstance(number, int) or number < 1 or number > 1000:
        raise ValueError("Input must be a positive integer between 1 and 1000.")
    
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "m", "cm", "d", "cd",
        "c", "xc", "l", "xl",
        "x", "ix", "v", "iv",
        "i"
        ]
    roman_num = ''
    i = 0
    while  number > 0:
        for _ in range(number // val[i]):
            roman_num += syb[i]
            number -= val[i]
        i += 1
    return roman_num.lower()
