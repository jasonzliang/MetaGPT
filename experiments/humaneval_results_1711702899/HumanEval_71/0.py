
def triangle_area(a, b, c):
    """Calculate the area of a triangle given its three sides.

    Args:
        a: The length of side a.
        b: The length of side b.
        c: The length of side c.

    Returns:
        The area of the triangle if it is valid, -1 otherwise.

    Raises:
        TypeError: If any of the parameters are not numbers.
        ValueError: If any of the parameters are negative or zero.
    """
    
    # Checking if all inputs are positive numbers
    for i in [a, b, c]:
        if not isinstance(i, (int, float)) or i <= 0:
            raise TypeError("All sides must be positive numbers")
            
    # Checking if the triangle inequality holds
    if a + b <= c or a + c <= b or b + c <= a:
        return -1
    
    # Calculating semi-perimeter
    s = (a + b + c) / 2
    
    # Using Heron's formula to calculate the area
    area = round((s * (s - a) * (s - b) * (s - c)) ** 0.5, 2)
    
    return area
