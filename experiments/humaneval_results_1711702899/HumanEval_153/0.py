
def Strongest_Extension(class_name, extensions):
    """Finds the strongest extension of a given class.

    Args:
        class_name: The name of the class (a string).
        extensions: A list of extensions to be used to load additional classes to the class.

    Returns:
        A string in this format: ClassName.StrongestExtensionName.

    Raises:
        TypeError: If class_name is not a string or if extensions is not a list.
        ValueError: If class_name is an empty string or if extensions is an empty list.
    """
    # Checking the types of inputs
    if not isinstance(class_name, str):
        raise TypeError("Class name must be a string.")
    if not isinstance(extensions, list):
        raise TypeError("Extensions must be a list.")
    
    # Checking for empty strings or empty lists
    if class_name == "":
        raise ValueError("Class name cannot be an empty string.")
    if len(extensions) == 0:
        raise ValueError("Extensions list cannot be empty.")
        
    # Calculating the strength of each extension and finding the strongest one
    strongest_extension = extensions[0]
    max_strength = sum(1 for c in strongest_extension if c.isupper()) - \
                  sum(1 for c in strongest_extension if c.islower())
    
    for ext in extensions:
        strength = sum(1 for c in ext if c.isupper()) - \
                   sum(1 for c in ext if c.islower())
        
        if strength > max_strength:
            strongest_extension = ext
            max_strength = strength
    
    return class_name + "." + strongest_extension
