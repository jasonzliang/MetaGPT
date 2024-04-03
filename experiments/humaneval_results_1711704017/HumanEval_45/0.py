
def triangle_area(base, height):
    """Given length of a base and high return area for a triangle.
    >>> triangle_area(5, 3)
    7.5
    """
    # Check if inputs are valid
    if not isinstance(base, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("Both base and height must be numbers.")
    elif base < 0 or height < 0:
        raise ValueError("Base and height must be non-negative.")
    
    # Calculate area using formula for triangle area
    return 0.5 * base * height
