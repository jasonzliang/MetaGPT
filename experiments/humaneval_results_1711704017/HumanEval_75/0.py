
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_multiply_prime(a):
    """Check if a number can be expressed as the product of three prime numbers."""
    for i in range(2, a//3 + 1):  # iterate from 2 to a/3
        if a % i == 0:  # if a is divisible by i
            if is_prime(i) and is_prime(a // (i * i)) and is_prime(a // i):  # check if factors are prime
                return True
    return False
