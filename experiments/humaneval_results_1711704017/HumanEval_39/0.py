
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def fibonacci(n):
    """Generate the Fibonacci sequence up to n-th number."""
    seq = [0, 1]
    while len(seq) < n + 1:
        seq.append(seq[-1] + seq[-2])
    return seq[n]

def prime_fib(n):
    """Return the n-th number that is a Fibonacci number and it's also prime."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input must be a positive integer.")
    
    count = 0
    i = 0
    while count < n:
        fib = fibonacci(i)
        if is_prime(fib):
            count += 1
        i += 1
        
    return fib
