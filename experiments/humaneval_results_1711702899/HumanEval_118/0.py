
def get_closest_vowel(word):
    """Finds the closest vowel that stands between two consonants from the right side of a word.

    Args:
        word: The input string to search for the closest vowel.

    Returns:
        str: The closest vowel found or an empty string if no vowels are found.

    Raises:
        TypeError: If the provided argument is not a string.
    """
    
    # Check that word is a string
    if not isinstance(word, str):
        raise TypeError("The input must be a string.")
        
    vowels = "aeiouAEIOU"
    for i in range(len(word) - 1, 0, -1):
        if word[i] in vowels and word[i-1] not in vowels and word[i+1] not in vowels:
            return word[i]
    
    # If no vowel is found
    return ""
