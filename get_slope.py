from timer import timer


@timer
def get_slope(x: list, y: list) -> float:
    import jax.numpy as np
    import jax.lax as lax

    sum_xy: int = 0
    sum_xx: int = 0

    x_bar: float = np.mean(x)
    y_bar: float = np.mean(y)

    def diff(_, pair):
        x, y = pair
        return None, (x - x_bar) * (y - y_bar)

    _, diff_output = lax.scan(diff, None, (x, y))
    sum_xy = np.sum(diff_output)

    def same(_, x):
        return None, (x - x_bar) * (x - x_bar)

    _, diff_output = lax.scan(same, None, x)
    sum_xx = np.sum(diff_output)

    return sum_xy / sum_xx


# import jax.numpy as np
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 4, 6, 8, 10])

# print(get_slope(x, y))  # should be 2
