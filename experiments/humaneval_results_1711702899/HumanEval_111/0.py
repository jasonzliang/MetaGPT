
def histogram(text):
    """Given a string representing a space separated lowercase letters, 
    return a dictionary of the letter with the most repetition and containing the corresponding count.
    If several letters have the same occurrence, return all of them.
    
    Args:
        text (str): A string of lowercase letters separated by spaces.
        
    Returns:
        dict: A dictionary where keys are letters from the input string 
              and values are their corresponding counts in the string.
              
    Raises:
        ValueError: If the input is not a string or contains non-lowercase letters.
    
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
        
    if not text.replace(' ', '').isalpha() or not text.replace(' ', '').islower():
        raise ValueError("Input can only contain lowercase letters and spaces.")
    
    letter_counts = {}
    for letter in text.split():
        if letter in letter_counts:
            letter_counts[letter] += 1
        else:
            letter_counts[letter] = 1
            
    max_count = max(letter_counts.values(), default=0)
    
    return {k: v for k, v in letter_counts.items() if v == max_count}
