
def iscube(a):
    """Checks if a number is a cube of some integer.

    Args:
        a: The integer to check.

    Returns:
        True if the number is a cube, False otherwise.

    Raises:
        No exceptions are raised in this function.
    """
    return round(a ** (1./3)) ** 3 == a
