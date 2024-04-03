
def triangle_area(side_length: float, height: float) -> float:
    """Calculate the area of a triangle given its side length and height.

    Args:
        side_length: The length of one side of the triangle.
        height: The perpendicular distance between the side and the line from the vertex to the opposite side.

    Returns:
        The area of the triangle.

    Raises:
        ValueError: If either `side_length` or `height` is less than zero.
    """
    if side_length < 0 or height < 0:
        raise ValueError("Both side length and height must be non-negative.")
    
    return 0.5 * side_length * height
