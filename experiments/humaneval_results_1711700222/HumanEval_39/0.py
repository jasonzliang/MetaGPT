
def is_prime(n):
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def prime_fib(n):
    count = 0
    for f in fibonacci():
        if is_prime(f) and f != 0:
            count += 1
            if count == n:
                return f
