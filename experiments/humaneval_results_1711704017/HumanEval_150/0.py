
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def x_or_y(n, x, y):
    """Return the value of x if n is a prime number and the value of y otherwise."""
    try:
        # Check if n is a prime number
        if is_prime(n):
            return x
        else:
            return y
    except TypeError:
        print("Invalid input. Please enter integers only.")
```

# Example usage:
print(x_or_y(7, 34, 12)) # Outputs: 34
print(x_or_y(15, 8, 5)) # Outputs: 5
