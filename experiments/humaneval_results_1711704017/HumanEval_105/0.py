
def by_length(arr):
    """
    Given an array of integers, sort the integers that are between 1 and 9 inclusive,
    reverse the resulting array, and then replace each digit by its corresponding name from
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine".
    """
    
    # Define a dictionary to map numbers to their names
    num_to_name = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
    
    # Filter the array to only include numbers between 1 and 9 inclusive, sort it, reverse it, and map each number to its name
    result = [num_to_name[i] for i in sorted([i for i in arr if 1 <= i <= 9], reverse=True)]
    
    return result
```

# Example usage:
# print(by_length([2, 1, 1, 4, 5, 8, 2, 3]))  # Outputs: ['Eight', 'Five', 'Four', 'Three', 'Two', 'Two', 'One', 'One']
```python
print(by_length([1, -1 , 55]))  # Outputs: ['One']
