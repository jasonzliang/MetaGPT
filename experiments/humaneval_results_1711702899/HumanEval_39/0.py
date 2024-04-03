
def prime_fib(n: int):
    """
    Returns the n-th number in the sequence of Fibonacci numbers that is also a prime number.

    Args:
        n (int): The position of the desired number in the sequence.

    Returns:
        int: The n-th Fibonacci prime number.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    
    def is_prime(num):
        """Check if num is a prime number."""
        if num <= 1 or (num % 2 == 0 and num > 2): 
            return False
        for i in range(3, int(num**0.5) + 1, 2):
            if num % i == 0:
                return False
        return True
    
    def fibonacci():
        """Generate Fibonacci numbers."""
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
            
    count = 0
    for num in fibonacci():
        if is_prime(num):
            count += 1
            if count == n:
                return num
