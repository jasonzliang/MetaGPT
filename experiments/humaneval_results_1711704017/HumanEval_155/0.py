
def even_odd_count(num):
    """Given an integer. Return a tuple that has the number of even and odd digits respectively."""
    # Convert num to positive if it's negative
    num = abs(num)
    # Initialize counters for even and odd digits
    even_count = 0
    odd_count = 0
    
    while num > 0:
        digit = num % 10
        if digit % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
        # Remove the last digit from num
        num //= 10
        
    return (even_count, odd_count)
```

# Example usage:
print(even_odd_count(-12))  # Outputs: (1, 1)
print(even_odd_count(123))  # Outputs: (1, 2)
