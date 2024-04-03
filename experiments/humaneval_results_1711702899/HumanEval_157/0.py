
def right_angle_triangle(a, b, c):
    """
    Given the lengths of the three sides of a triangle, this function returns True if the 
    triangle is right-angled (i.e., one angle is right or 90 degrees), and False otherwise.

    Args:
        a: The length of side a in the triangle.
        b: The length of side b in the triangle.
        c: The length of side c in the triangle.

    Returns:
        True if the triangle is right-angled, False otherwise.

    Raises:
        ValueError: If any of the lengths are less than or equal to zero.
    """
    
    # Check for invalid inputs
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("The length of each side must be greater than 0.")
        
    # Sort the sides so that 'c' is the hypotenuse (longest side)
    a, b, c = sorted([a, b, c])
    
    # Check if triangle is right-angled using Pythagorean theorem
    return abs(a**2 + b**2 - c**2) < 1e-9  # Use a small tolerance for floating point errors
