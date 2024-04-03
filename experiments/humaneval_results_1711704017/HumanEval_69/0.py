
def search(lst):
    """
    Function to find the greatest integer greater than zero with a frequency 
    greater than or equal to its value.
    """
    # Create a dictionary to store the frequency of each number in the list
    freq_dict = {}
    for num in lst:
        if num not in freq_dict:
            freq_dict[num] = 1
        else:
            freq_dict[num] += 1
    
    # Iterate over the dictionary to find the number with frequency >= its value
    max_val = -1
    for key, val in freq_dict.items():
        if key > max_val and val >= key:
            max_val = key
            
    return max_val
