import numpy as np

def transform_to_range_class(y):
    """Transforms discrete numbers into range-based classification."""
    return (np.array(y) - 1) // 25


if __name__ == "__main__":
    # Example usage
    y_train = [5, 10, 50, 55, 75, 100, 125, 150, 175, 200, 250]
    y_test = [2, 4, 8, 12, 14, 26, 28 ,32, 36, ]

    # Transforming values
    y_train_transformed = transform_to_range_class(y_train)
    y_test_transformed = transform_to_range_class(y_test)

    print("Transformed y_train:", y_train_transformed)
    print("Transformed y_test:", y_test_transformed)
