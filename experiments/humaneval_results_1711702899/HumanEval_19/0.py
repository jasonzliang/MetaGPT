
def sort_numbers(numbers: str) -> str:
    """Sort a space-delimited string of numberals from 'zero' to 'nine'.

    Args:
        numbers: A space-delimited string containing numberals from 'zero' to 'nine'.

    Returns:
        The sorted string of numberals.

    Raises:
        ValueError: If the input string contains a numeral not in the range 'zero' to 'nine'.
    """
    
    # Define a dictionary mapping number names to their corresponding integers
    number_dict = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9
    }
    
    # Split the input string into a list of number names
    numbers_list = numbers.split()
    
    # Check if all number names in the list are valid
    for numeral in numbers_list:
        if numeral not in number_dict:
            raise ValueError(f"Invalid numeral: {numeral}")
            
    # Sort the list of number names based on their corresponding integers
    sorted_numbers = sorted(numbers_list, key=lambda x: number_dict[x])
    
    # Join the sorted list back into a string and return it
    return " ".join(sorted_numbers)
