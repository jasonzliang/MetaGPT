
def largest_prime_factor(n: int) -> int:
    """Return the largest prime factor of n. Assume n >1 and is not a prime."""
    i = 2
    while i * i <= n:
        if n % i:  # If n is not divisible by i, increment i
            i += 1
        else:  # Otherwise, divide n by i as long as possible
            n //= i
    return n
