
def Strongest_Extension(class_name, extensions):
    def strength(ext):
        return sum(1 for c in ext if c.isupper()) - sum(1 for c in ext if c.islower())

    strongest = max(extensions, key=strength)
    return f'{class_name}.{strongest}'
