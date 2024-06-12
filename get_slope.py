import jax.numpy as np


def get_slope(x: list, y: list) -> float:
    sum_xy: int = 0
    sum_xx: int = 0

    x_bar: float = np.mean(x)
    y_bar: float = np.mean(y)

    for i in range(0, len(x) - 1):
        x_diff: float = x[i] - x_bar
        y_diff: float = y[i] - y_bar
        sum_xy += x_diff * y_diff

    for i in range(0, len(x) - 1):
        x_diff: float = x[i] - x_bar
        sum_xx += x_diff * x_diff

    return sum_xy / sum_xx


# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 4, 6, 8, 10])

# print(get_slope(x, y))  # should be 2
