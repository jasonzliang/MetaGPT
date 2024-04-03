
def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one',  'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest.
    >>> sort_numbers('three one five')
    'one three five'
    """
    # Define a dictionary that maps number words to their corresponding integers
    num_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    
    # Split the input string into a list of words
    num_list = numbers.split()
    
    # Convert each word to its corresponding integer using the dictionary
    num_list = [num_dict[word] for word in num_list]
    
    # Sort the list of integers
    num_list.sort()
    
    # Convert each integer back to its corresponding word using the dictionary
    sorted_words = [key for key, value in num_dict.items() if value == num for num in num_list]
    
    # Join the sorted words into a string with spaces between them
    return ' '.join(sorted_words)
