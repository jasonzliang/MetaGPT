
def largest_prime_factor(n: int) -> int:
    """Return the largest prime factor of n. Assume n >1 and is not a prime.

    Args:
        n: The number to find the largest prime factor for.

    Returns:
        The largest prime factor of n.

    Raises:
        ValueError: If n <= 1, as there are no prime factors for these numbers.
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")
    
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n
