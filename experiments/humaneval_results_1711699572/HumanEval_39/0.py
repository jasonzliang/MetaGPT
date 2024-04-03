def is_prime(n):
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def prime_fib(n):
    count = 0
    i = 0
    while count < n:
        fib = fibonacci(i)
        if is_prime(fib):
            count += 1
        i += 1
    return fib 