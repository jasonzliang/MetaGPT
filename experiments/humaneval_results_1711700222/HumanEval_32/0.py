
import math
def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def find_zero(xs: list):
    """ 
    xs are coefficients of a polynomial.
    find_zero finds x such that poly(x) = 0.
    find_zero returns only one zero point, even if there are many.
    Moreover, find_zero only takes lists xs having an even number of coefficients
    and the largest non-zero coefficient as it guarantees a solution.
    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
    n = len(xs)
    if n % 2 == 0:
        raise ValueError("The number of coefficients must be odd.")
    max_coeff = max([abs(c) for c in xs])
    x = 0.5 * max_coeff
    while True:
        fx = poly(xs, x)
        if abs(fx) < 1e-6: # Stop when the function value is small enough
            return round(x, 2)
        dfdx = sum([i*c*math.pow(x, i-1) for i, c in enumerate(xs[1:], start=1)])
        if abs(dfdx) < 1e-6: # Stop when the derivative is small enough
            raise ValueError("Cannot find a zero point.")
        x = x - fx / dfdx
