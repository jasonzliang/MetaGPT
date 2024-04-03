
def words_in_sentence(sentence):
    """
    This function takes a sentence as input and returns a string containing the words from the original sentence whose lengths are prime numbers. 

    Args:
        sentence (str): A string representing a sentence, with words separated by spaces.

    Returns:
        str: A string that contains the words from the original sentence whose lengths are prime numbers. The order of the words in the new string is the same as the original one.

    Raises:
        ValueError: If the length of the sentence exceeds 100 characters or if it contains non-alphabetical characters.
        
    """
    
    def is_prime(n):
        """
        This function checks whether a number is prime or not.

        Args:
            n (int): The number to be checked.

        Returns:
            bool: True if the number is prime, False otherwise.
            
        """
        
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # Check constraints
    if len(sentence) > 100 or not sentence.replace(' ', '').isalpha():
        raise ValueError("Invalid input")
        
    words = sentence.split()
    prime_words = [word for word in words if is_prime(len(word))]
    
    return " ".join(prime_words)
