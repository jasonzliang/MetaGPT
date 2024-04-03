
def derivative(coefficients: list) -> list:
    """Calculate the derivative of a polynomial represented by coefficients.

    Args:
        coefficients: A list of coefficients where index represents power 
                       and value at that index represents coefficient for x^index.

    Returns:
        The derivative of the polynomial as a list of coefficients.

    Raises:
        TypeError: If input is not a list.
        ValueError: If any element in the list is not an integer or float.
    
    Examples:
        >>> derivative([3, 1, 2, 4, 5])
        [1, 4, 12, 20]
        >>> derivative([1, 2, 3])
        [2, 6]
    """
    
    # Check if input is a list
    if not isinstance(coefficients, list):
        raise TypeError("Input should be a list")
    
    # Check if all elements in the list are integers or floats
    for i in coefficients:
        if not isinstance(i, (int, float)):
            raise ValueError("All elements in the list should be integers or floats")
            
    derivative_coefficients = [i*val for i, val in enumerate(coefficients)][1:]
    
    return derivative_coefficients
