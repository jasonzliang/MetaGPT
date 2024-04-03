
from typing import List

def factorize(n: int) -> List[int]:
    """Return list of prime factors of given integer in the order from smallest to largest.
    Each of the factors should be listed number of times corresponding to how many times it appears in factorization.
    Input number should be equal to the product of all factors.
    
    Args:
        n: The integer to be factorized.
        
    Returns:
        A list of prime factors of the input integer, sorted from smallest to largest.
        
    Raises:
        ValueError: If the input is not a positive integer.
    """
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    factors = []
    i = 2  # Start from smallest prime number
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
            
    if n > 1:
        factors.append(n)
        
    return factors
