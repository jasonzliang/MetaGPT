
def find_max(words):
    """Write a function that accepts a list of strings.
    The list contains different words. Return the word with maximum number
    of unique characters. If multiple strings have maximum number of unique
    characters, return the one which comes first in lexicographical order."""
    
    # Create an empty dictionary to store each word and its unique character count
    word_dict = {}
    
    for word in words:
        # Convert each word into a set to remove duplicates and then get the length of this set
        word_dict[word] = len(set(word))
        
    # Find the maximum value in the dictionary
    max_value = max(word_dict.values())
    
    # Create a new dictionary with words that have the maximum unique character count
    max_words = {k: v for k, v in word_dict.items() if v == max_value}
    
    # Return the first word (in lexicographical order) from this dictionary
    return sorted(max_words.keys())[0]
