
import math
from typing import List

def poly(xs: List[float], x: float) -> float:
    """Evaluates polynomial with coefficients xs at point x.

    Args:
        xs: A list of coefficients for the polynomial.
        x: The value at which to evaluate the polynomial.

    Returns:
        The result of evaluating the polynomial at x.
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def find_zero(xs: List[float]) -> float:
    """Finds a zero of a polynomial with coefficients xs.

    Args:
        xs: A list of coefficients for the polynomial. The number of coefficients must be even, and the last non-zero coefficient is used to estimate the zero point.

    Returns:
        A zero of the polynomial.

    Raises:
        ValueError: If the number of coefficients is not even or if there are no non-zero coefficients.
    """
    # Check that the number of coefficients is even and there's at least one non-zero coefficient
    if len(xs) % 2 == 0 and any(coeff != 0 for coeff in xs):
        # Use the largest non-zero coefficient to estimate a zero point
        max_coeff = max([abs(coeff) for coeff in xs])
        return -max_coeff / xs[-1]
    else:
        raise ValueError("The number of coefficients must be even and there must be at least one non-zero coefficient.")
