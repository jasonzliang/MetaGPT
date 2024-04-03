
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_length(string):
    """Check if the length of a string is a prime number."""
    return is_prime(len(string))
```

# Example usage:
print(prime_length('Hello'))  # Returns: True
print(prime_length('abcdcba'))  # Returns: True
print(prime_length('kittens'))  # Returns: True
print(prime_length('orange'))  # Returns: False
