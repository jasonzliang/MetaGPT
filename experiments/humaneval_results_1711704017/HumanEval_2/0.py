
def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    # Check if input is a positive floating point number
    if not isinstance(number, float) or number <= 0:
        raise ValueError("Input must be a positive floating point number")
        
    # Use the math module's modf function to separate the integer and decimal parts of the number
    import math
    _, decimals = math.modf(number)
    
    return decimals
