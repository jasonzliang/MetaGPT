
def find_max(words):
    """Find the word with maximum number of unique characters. If multiple strings have 
    maximum number of unique characters, return the one which comes first in lexicographical order.

    Args:
        words: A list of strings.

    Returns:
        The word with maximum number of unique characters or the first string if there are no words.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list contains non-string elements.
    """
    
    # Checking if the input is a list
    if not isinstance(words, list):
        raise TypeError("Input should be a list of strings")
        
    # Checking if all elements in the list are strings
    for word in words:
        if not isinstance(word, str):
            raise ValueError("All elements in the list should be strings")
    
    # Initializing max_unique and max_word to keep track of maximum unique characters and corresponding word
    max_unique = 0
    max_word = ""
    
    for word in words:
        # Finding number of unique characters in each word
        num_unique = len(set(word))
        
        if num_unique > max_unique or (num_unique == max_unique and word < max_word):
            max_unique = num_unique
            max_word = word
            
    return max_word
