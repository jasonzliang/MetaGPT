
def derivative(xs: list) -> list:
    """ xs represent coefficients of a polynomial.
    xs[0] + xs[1] * x + xs[2] * x^2 + ....
    Return derivative of this polynomial in the same form.
    >>> derivative([3, 1, 2, 4, 5])
    [1, 4, 12, 20]
    >>> derivative([1, 2, 3])
    [2, 6]
    """
    # Check if input is a list
    if not isinstance(xs, list):
        raise TypeError("Input must be a list")
        
    # Check if all elements in the list are integers or floats
    for i in xs:
        if not isinstance(i, (int, float)):
            raise ValueError("All elements of input list must be numbers")
    
    # Calculate derivative
    return [i*x for i, x in enumerate(xs)][1:]
