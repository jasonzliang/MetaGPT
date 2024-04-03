
def closest_integer(value):
    '''
    Create a function that takes a value (string) representing a number
    and returns the closest integer to it. If the number is equidistant
    from two integers, round it away from zero.

    Examples
    >>> closest_integer("10")
     10
    >>> closest_integer("15.3")
     15

    Note:
    Rounding away from zero means that if the given number is equidistant
    from two integers, the one you should return is the one that is the
    farthest from zero. For example closest_integer("14.5") should
    return 15 and closest_integer("-14.5") should return -15.
    '''
    
    # Convert string to float
    num = float(value)
    
    # Check if the number is positive or negative
    if num >= 0:
        # Round the number away from zero using Python's built-in round function
        return int(round(num))
    else:
        # For negative numbers, subtract the rounded absolute value of the number to get the closest integer
        return -int(round(abs(num)))
