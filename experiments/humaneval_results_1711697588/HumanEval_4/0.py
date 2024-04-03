
    from typing import List
    def mean_absolute_deviation(numbers: List[float]) -> float:
        n = len(numbers)
        if n == 0:
            return 0.0
        mean = sum(numbers) / n
        absolute_differences = [abs(x - mean) for x in numbers]
        mad = sum(absolute_differences) / n
        return mad
    