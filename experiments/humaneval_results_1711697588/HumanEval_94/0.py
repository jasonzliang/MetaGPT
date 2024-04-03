
def is_prime(n):
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def largest_prime_sum(lst):
    primes = [x for x in lst if is_prime(x)]
    if not primes:
        return None
    max_prime = max(primes)
    return sum(int(digit) for digit in str(max_prime))
