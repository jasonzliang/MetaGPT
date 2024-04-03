
def calculate_strength(extension):
    """Calculate the strength of an extension."""
    cap = sum(1 for c in extension if c.isupper())
    sm = sum(1 for c in extension if c.islower())
    return cap - sm

def strongest_extension(class_name, extensions):
    """Find the strongest extension and return a string in the format ClassName.StrongestExtensionName."""
    if not class_name or not extensions:
        raise ValueError("Class name and extensions must be provided.")
    
    # Calculate strength for each extension
    strengths = {ext: calculate_strength(ext) for ext in extensions}
    
    # Find the strongest extension
    strongest = max(extensions, key=lambda x: strengths[x])
    
    return f"{class_name}.{strongest}"
```

# Example usage:
print(strongest_extension('Slices', ['SErviNGSliCes', 'Cheese', 'StuFfed']))  # Outputs: Slices.SErviNGSliCes
