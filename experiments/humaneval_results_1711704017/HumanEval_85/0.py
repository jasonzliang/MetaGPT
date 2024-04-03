
def add_even_elements_at_odd_indices(lst):
    """Given a non-empty list of integers lst, add the even elements that are at odd indices."""
    if not isinstance(lst, list) or len(lst) == 0:
        raise ValueError("Input should be a non-empty list")
    
    result = 0
    for i in range(len(lst)):
        # Check if index is odd and element at that index is even
        if i % 2 != 0 and lst[i] % 2 == 0:
            result += lst[i]
    
    return result

# Example usage:
print(add_even_elements_at_odd_indices([4, 2, 6, 7])) # Outputs: 2
