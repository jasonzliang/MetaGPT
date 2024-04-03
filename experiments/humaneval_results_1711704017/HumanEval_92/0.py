
def any_int(x, y, z):
    """
    Check if one of the numbers is equal to the sum of the other two.
    All numbers must be integers.
    """
    # Ensure all inputs are integers
    try:
        x = int(x)
        y = int(y)
        z = int(z)
    except ValueError:
        return False
    
    # Check if one of the numbers is equal to the sum of the other two
    if (x == y + z) or (y == x + z) or (z == x + y):
        return True
    else:
        return False
```

# Example usage:
# print(any_int(5, 2, 7))  # Returns: True
# print(any_int(3, 2, 2))  # Returns: False
# print(any_int(3, -2, 1))  # Returns: True
# print(any_int(3.6, -2.2, 2))  # Returns: False
