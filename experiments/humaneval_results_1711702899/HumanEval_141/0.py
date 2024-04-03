
def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns 'Yes' if the 
    file's name is valid, and returns 'No' otherwise.

    Args:
        file_name (str): The name of the file to be checked.

    Returns:
        str: 'Yes' if the file's name is valid, 'No' otherwise.

    Raises:
        TypeError: If the input argument is not a string.
    """
    
    # Check if the input is a string
    if not isinstance(file_name, str):
        raise TypeError("The input should be a string.")
        
    # Split the file name into two parts at the dot
    try:
        name, extension = file_name.rsplit('.', 1)
    except ValueError:
        return 'No'
    
    # Check if there are more than three digits in the file's name
    if sum(c.isdigit() for c in file_name) > 3:
        return 'No'
        
    # Check if the substring before the dot starts with a letter from the latin alphabet
    if not name[0].isalpha():
        return 'No'
    
    # Check if the substring after the dot is one of these: ['txt', 'exe', 'dll']
    if extension not in ['txt', 'exe', 'dll']:
        return 'No'
        
    return 'Yes'
