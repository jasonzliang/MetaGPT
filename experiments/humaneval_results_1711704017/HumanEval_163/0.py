
def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.
    """
    # Ensure inputs are positive integers
    if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b < 0:
        raise ValueError("Both inputs must be positive integers.")
    
    # Swap the numbers if a > b
    if a > b:
        a, b = b, a
        
    result = []
    for num in range(a, b + 1):
        if num % 2 == 0:
            result.append(num)
            
    return result
```
This function first checks that the inputs are positive integers and raises an error if they are not. It then swaps a and b if a > b to ensure the range is correct. The function then iterates over each number in this range, checking if it's even (i.e., its remainder when divided by 2 is zero). If so, it adds it to the result list. Finally, it returns the result list.

Example usage:
```python
print(generate_integers(2, 8)) # Output: [2, 4, 6, 8]
print(generate_integers(8, 2)) # Output: [2, 4, 6, 8]
print(generate_integers(10, 14)) # Output: [10, 12, 14]
