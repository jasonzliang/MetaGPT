
def is_prime(n):
    """Return true if a given number is prime, and false otherwise."""
    # Check for invalid inputs
    if type(n) != int or n < 2:
        return False
    # 2 and 3 are prime numbers
    elif n == 2 or n == 3:
        return True
    # Eliminate even numbers
    elif n % 2 == 0:
        return False
    else:
        # Check divisibility up to the square root of n
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
