
def remove_vowels(text):
    """
    This function takes a string and returns the same string without vowels.
    """
    # Define the vowels
    vowels = 'aeiouAEIOU'

    # Use list comprehension to create a new string that excludes vowels
    text_without_vowels = ''.join([char for char in text if char not in vowels])
    
    return text_without_vowels
