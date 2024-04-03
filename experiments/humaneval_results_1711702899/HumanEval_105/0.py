
def by_length(arr):
    """
    Given an array of integers, sort the integers that are between 1 and 9 inclusive,
    reverse the resulting array, and then replace each digit by its corresponding name from
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine".

    Args:
        arr: An array of integers.

    Returns:
        A list of strings representing the sorted, reversed and replaced digits.

    Raises:
        TypeError: If the input is not a list or if it contains non-integer elements.
    """
    
    # Checking if arr is a list
    if not isinstance(arr, list):
        raise TypeError("The input should be a list.")
        
    # Filtering out integers between 1 and 9 inclusive
    filtered_arr = [i for i in arr if isinstance(i, int) and 1 <= i <= 9]
    
    # Sorting the array in ascending order
    sorted_arr = sorted(filtered_arr)
    
    # Reversing the array
    reversed_arr = list(reversed(sorted_arr))
    
    # Converting each digit to its corresponding name
    num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
    replaced_arr = [num2words[i] for i in reversed_arr]
    
    return replaced_arr
