
def modp(n: int, p: int) -> int:
    """Return 2^n modulo p (be aware of numerics).

    Args:
        n: The exponent for the base 2.
        p: The modulus to be used in the calculation.

    Returns:
        The result of 2^n modulo p.

    Raises:
        ValueError: If either n or p is less than zero.

    >>> modp(3, 5)
    3
    >>> modp(101, 101)
    2
    >>> modp(0, 101)
    1
    >>> modp(3, 11)
    8
    >>> modp(100, 101)
    1
    """
    if n < 0 or p < 0:
        raise ValueError("Both n and p must be non-negative.")
    return pow(2, n, p)
