
def sum_of_digits(n):
    """Calculate the sum of digits of a number.

    Args:
        n: The integer to calculate the sum of its digits.

    Returns:
        The sum of the digits of the input number.
    """
    return sum(int(digit) for digit in str(n))

def is_prime(n):
    """Check if a number is prime.

    Args:
        n: The integer to check if it's a prime number.

    Returns:
        True if the input number is prime, False otherwise.
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def largest_prime_sum(lst):
    """Find the sum of digits of the largest prime number in a list.

    Args:
        lst: The list of integers to search for the largest prime number.

    Returns:
        The sum of the digits of the largest prime number in the input list, or 0 if no prime numbers are found.
    """
    primes = [x for x in lst if is_prime(x)]
    return sum_of_digits(max(primes)) if primes else 0
