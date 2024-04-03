
def words_string(s):
    """
    Split the string into words and return an array of the words.
    """
    try:
        # Replace commas with spaces for splitting
        s = s.replace(',', ' ')
        
        # Split the string into words
        words = s.split()
        
        return words
    
    except Exception as e:
        print("An error occurred: ", str(e))
