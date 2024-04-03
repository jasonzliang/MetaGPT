
def double_the_difference(lst):
    """
    Given a list of numbers, return the sum of squares of the numbers in the list that are odd. 
    Ignore numbers that are negative or not integers.
    """
    if not lst: # If the input list is empty, return 0.
        return 0
    
    result = 0
    for num in lst:
        try:
            if num < 0 or num != int(num): # Ignore numbers that are negative or not integers.
                continue
            elif num % 2 == 1: # If the number is odd, square it and add to result.
                result += num ** 2
        except TypeError as e: # Handle invalid inputs (non-numeric types).
            print(f"Invalid input: {e}")
    return result
```

# Example usage:
print(double_the_difference([1, 3, 2, 0])) # Outputs: 10
print(double_the_difference([-1, -2, 0])) # Outputs: 0
print(double_the_difference([9, -2])) # Outputs: 81
print(double_the_difference([0])) # Outputs: 0  
