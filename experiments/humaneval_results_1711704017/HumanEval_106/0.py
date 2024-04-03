
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

def f(n):
    """Implement the function f that takes n as a parameter, and returns a list of size n, such that 
       the value of the element at index i is the factorial of i if i is even or the sum of numbers from 1 to i otherwise.
       i starts from 1."""
    result = []
    for i in range(1, n + 1):
        if i % 2 == 0: # If i is even
            result.append(factorial(i))
        else: # If i is odd
            sum_of_numbers = 0
            for j in range(1, i + 1):
                sum_of_numbers += j
            result.append(sum_of_numbers)
    return result
```

# Example usage:
print(f(5))  # Outputs: [1, 2, 6, 24, 15]
