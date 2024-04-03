
def count_boredoms(s):
    """Count the number of boredoms in a string.

    A boredom is defined as any sentence that starts with the word "I". 
    Sentences are delimited by '.', '?' or '!'.

    Args:
        s: The input string to be checked for boredoms.

    Returns:
        An integer representing the number of boredoms in the string.

    Raises:
        TypeError: If the input is not a string.
    """
    
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")
        
    # Split the string into sentences based on '.', '?', and '!'
    sentences = [sentence.strip() for sentence in re.split('[.!?]', s) if sentence]
    
    # Count the number of sentences that start with "I"
    boredoms = sum(1 for sentence in sentences if sentence.startswith("I"))
    
    return boredoms
