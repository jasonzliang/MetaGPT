
def histogram(test_string):
    """Given a string representing a space separated lowercase letters, 
       return a dictionary of the letter with the most repetition and containing the corresponding count.
       If several letters have the same occurrence, return all of them."""
    
    # Split the input string into words
    words = test_string.split()
    
    # Create an empty dictionary to store the frequency of each word
    freq_dict = {}
    
    for word in words:
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1
            
    # Find the maximum frequency
    max_freq = max(freq_dict.values(), default=0)
    
    # Create a new dictionary with only the words that have the maximum frequency
    result_dict = {k: v for k, v in freq_dict.items() if v == max_freq}
    
    return result_dict
