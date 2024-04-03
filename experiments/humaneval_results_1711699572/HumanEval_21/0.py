
    def rescale_to_unit(numbers):
        min_val = min(numbers)
        max_val = max(numbers)
        range_val = max_val - min_val
        return [(x - min_val) / range_val for x in numbers]
    