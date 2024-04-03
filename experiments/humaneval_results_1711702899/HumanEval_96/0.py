
def count_up_to(n):
    """Implement a function that takes an non-negative integer and returns an array of the first n 
    integers that are prime numbers and less than n.

    Args:
        n: The upper limit for the prime number search (non-negative integer).

    Returns:
        A list of all prime numbers less than 'n'. If 'n' is less than or equal to 1, returns an empty list.

    Raises:
        TypeError: If 'n' is not an integer.
        ValueError: If 'n' is negative.
    """
    
    if not isinstance(n, int):
        raise TypeError("Input must be a non-negative integer.")
    elif n < 0:
        raise ValueError("Input must be a non-negative integer.")
        
    primes = []
    for possible_prime in range(2, n):
        is_prime = True
        for num in range(2, possible_prime):
            if possible_prime % num == 0:
                is_prime = False                
        if is_prime:
            primes.append(possible_prime)
    return primes
