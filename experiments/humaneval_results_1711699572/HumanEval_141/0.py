
import re

def file_name_check(file_name):
    pattern = r'^[a-zA-Z][a-zA-Z0-9]{0,2}\.(txt|exe|dll)$'
    if len(re.findall('\d', file_name)) > 3:
        return 'No'
    elif re.match(pattern, file_name):
        return 'Yes'
    else:
        return 'No'
