
def is_multiply_prime(a):
    primes = [2, 3, 5, 7]
    factors = []
    for prime in primes:
        while a % prime == 0:
            factors.append(prime)
            a /= prime
    return len(factors) == 3 and all(factors.count(i) == [2, 3, 5].count(i) for i in set(factors))
    ```
    ```python
print(is_multiply_prime(30)) # True
print(is_multiply_prime(15)) # False
print(is_multiply_prime(210)) # True
print(is_multiply_prime(70)) # False
