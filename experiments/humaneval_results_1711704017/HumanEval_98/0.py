
def count_upper(s):
    """
    Given a string s, count the number of uppercase vowels in even indices.

    For example:
    count_upper('aBCdEf') returns 1
    count_upper('abcdefg') returns 0
    count_upper('dBBE') returns 0
    """
    # Define the set of uppercase vowels
    vowels = {'A', 'E', 'I', 'O', 'U'}
    
    # Initialize a counter for the number of uppercase vowels in even indices
    count = 0
    
    # Iterate over the string, checking every other character (even indices)
    for i in range(0, len(s), 2):
        if s[i] in vowels:
            count += 1
            
    return count
