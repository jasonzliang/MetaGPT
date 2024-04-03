
def is_multiply_prime(a):
    primes = [2, 3, 5, 7]
    factors = []
    for prime in primes:
        while a % prime == 0:
            factors.append(prime)
            a /= prime
    return len(factors) == 3 and all(factors.count(i) == [2, 3, 5].count(i) for i in set(factors))
