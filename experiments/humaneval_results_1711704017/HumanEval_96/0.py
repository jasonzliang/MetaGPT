
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    elif n == 2:
        return True
    else:
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

def count_up_to(n):
    """Return an array of the first n integers that are prime numbers and less than n."""
    primes = []
    for i in range(2, n):  # Start from 2 because 1 is not a prime number
        if len(primes) == n:
            break
        elif is_prime(i):
            primes.append(i)
    return primes
