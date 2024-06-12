from app import main

if __name__ == "__main__":
    import jax.numpy as jnp

    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([3,6,7,8,9])

    func, loss = main(x, y)

    fahad = func(6)

    print(f"Result: {fahad} | Loss: {loss}")
