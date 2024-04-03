
def factorial(n):
    """Calculate the factorial of a given number n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

def special_factorial(n):
    """Calculate the Brazilian factorial of a given number n."""
    if n < 0:
        raise ValueError("Special Factorial is not defined for negative numbers.")
    else:
        result = 1
        for i in range(1, n + 1):
            result *= factorial(i)
        return result
```

# Example usage:
print(special_factorial(4))  # Outputs: 288
