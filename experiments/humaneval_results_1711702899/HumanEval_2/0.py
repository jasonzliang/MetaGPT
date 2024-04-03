
def truncate_number(number: float) -> float:
    """Given a positive floating point number, it can be decomposed into an integer part 
    (largest integer smaller than given number) and decimals (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    if not isinstance(number, float):
        raise ValueError("Input must be a floating point number.")
    elif number < 0:
        raise ValueError("Input must be a positive number.")
        
    integer_part = int(number)
    decimal_part = number - integer_part
    
    return decimal_part
