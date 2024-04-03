
def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns 'Yes' if the file's name is valid, and returns 'No' otherwise."""
    # Split the filename into two parts at the dot.
    try:
        name, extension = file_name.rsplit('.', 1)
    except ValueError:
        return 'No'
    
    # Check if there are more than three digits in the filename.
    if sum(c.isdigit() for c in name) > 3:
        return 'No'
    
    # Check if the filename starts with a letter from the latin alphabet.
    if not name[0].isalpha():
        return 'No'
    
    # Check if the extension is one of these: ['txt', 'exe', 'dll'].
    if extension not in ['txt', 'exe', 'dll']:
        return 'No'
    
    return 'Yes'
