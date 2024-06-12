from app import main
from timeit import timeit

if __name__ == "__main__":
    import jax.numpy as jnp

    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([2, 4, 6, 8, 10])

    func = main(x, y)

    fahad = func(6)
    time = timeit(func(6), number=10000)


    print(f"Result: {fahad} | Expected: 12")
    print(f"Runtime: {time}")


