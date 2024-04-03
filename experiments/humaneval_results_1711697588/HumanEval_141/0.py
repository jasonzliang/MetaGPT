
def file_name_check(file_name):
    # Split the filename into name and extension
    name, ext = file_name.split('.')
    
    # Check if there are more than 3 digits in the name
    if sum(c.isdigit() for c in name) > 3:
        return 'No'
        
    # Check if the name starts with a letter from the latin alphabet
    if not name[0].isalpha():
        return 'No'
    
    # Check if the extension is one of ['txt', 'exe', 'dll']
    if ext not in ['txt', 'exe', 'dll']:
        return 'No'
        
    return 'Yes'
