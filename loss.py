from timer import timer


@timer
def mse(y_acc: list, y_pred: list):
    import jax.numpy as np
    import jax.lax as lax

    n: int = len(y_acc)

    # MSE = (1/n) * Σ(yᵢ - ȳ)²

    def scanned_fun(_, pair):
        x, y = pair
        return None, (x - y) * (x - y)

    _, zs = lax.scan(scanned_fun, None, (y_acc, y_pred))

    return (1 / n) * np.sum(zs)


# import jax.numpy as np

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([1, 2, 3, 4.3452, 5.6453])
# print(mse(x, y))
