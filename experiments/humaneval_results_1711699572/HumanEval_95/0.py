
def check_dict_case(d):
    if not d:  # If the dictionary is empty
        return False
    case = None
    for key in d.keys():
        if type(key) != str:  # If a non-string key is found
            return False
        if case is None:  # Set the case based on the first key
            case = key.islower()
        elif (case and not key.islower()) or (not case and not key.isupper()):  # If the case doesn't match
            return False
    return True
