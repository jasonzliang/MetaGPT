
def is_multiply_prime(n):
    """Check if a number can be expressed as the product of exactly 3 prime numbers.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number can be expressed as the product of exactly 3 prime numbers, False otherwise.

    Raises:
        ValueError: If input is not an integer or less than 2.
    """
    # Checking if n is an integer and greater than 1
    if not isinstance(n, int) or n < 2:
        raise ValueError("Input must be an integer greater than 1.")
    
    # Function to check if a number is prime
    def is_prime(num):
        for i in range(2, num):
            if (num % i) == 0:
                return False
        return True

    primes = [i for i in range(2, n+1) if is_prime(i)]
    
    # Checking if the number can be expressed as a product of exactly 3 prime numbers
    count = 0
    for p1 in primes:
        if n % p1 == 0:
            quotient = n // p1
            for p2 in primes:
                if quotient % p2 == 0 and is_prime(quotient // p2):
                    count += 1
    
    return count == 3
