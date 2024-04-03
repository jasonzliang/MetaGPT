
def odd_count(lst):
    """Given a list of strings, where each string consists of only digits, return a list.
    Each element i of the output should be "the number of odd elements in the
    string i of the input." where all the i's should be replaced by the number
    of odd digits in the i't string of the input."""
    
    def count_odd(string):
        """Count the number of odd numbers in a given string."""
        return sum([int(char) % 2 for char in string])
    
    result = []
    for i, string in enumerate(lst, start=1):
        if not string.isdigit():
            raise ValueError("All strings must consist of digits.")
        
        odd_numbers = count_odd(string)
        result.append(f"the number of odd elements {odd_numbers}n the str{odd_numbers}ng {odd_numbers} of the {odd_numbers}nput.")
    
    return result
