
def select_words(s, n):
    """Given a string s and a natural number n, this function returns a list of all words 
    from string s that contain exactly n consonants, in order these words appear in the 
    string s. If the string s is empty then the function should return an empty list.
    
    Args:
        s: A string containing only letters and spaces.
        n: A natural number representing the exact number of consonants a word must have.
        
    Returns:
        A list of words from string s that contain exactly n consonants, in order these 
        words appear in the string s. If no such words exist, returns an empty list.
    
    Raises:
        TypeError: If s is not a string or if n is not a natural number.
        
    """
    # Checking if inputs are of correct type
    if not isinstance(s, str):
        raise TypeError("The first argument must be a string.")
    if not isinstance(n, int) or n < 0:
        raise TypeError("The second argument must be a natural number.")
    
    # Defining the set of consonants
    consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
    
    # Splitting the string into words and filtering out those with n consonants
    return [word for word in s.split() if sum(ch in consonants for ch in word) == n]
