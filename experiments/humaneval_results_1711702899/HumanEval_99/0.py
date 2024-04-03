
def closest_integer(value):
    """This function takes a string representing a number and returns the closest integer to it. 
    If the number is equidistant from two integers, it rounds away from zero.

    Args:
        value (str): A string representation of a floating point number.

    Returns:
        int: The closest integer to the input value.

    Raises:
        ValueError: If the input is not a valid floating point number.
    """
    try:
        num = float(value)
        if num > 0:
            return int(num + 0.5)
        else:
            return int(num - 0.5)
    except ValueError as e:
        raise ValueError("The input value is not a valid floating point number.") from e
