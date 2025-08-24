import numpy as np

class SimpleTrainer:
    """A minimalistic trainer to simulate weight updates."""

    def __init__(self, input_dim: int = 10):
        self.weights = np.random.randn(input_dim).tolist()

    def train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Simulate a single training step and update weights.
        (Replace with real ML logic later)
        """
        # Simple simulation: gradient = mean error
        preds = X.dot(np.array(self.weights))
        error = preds - y
        grad = X.T.dot(error) / len(y)

        # Simple SGD update
        lr = 0.01
        self.weights = (np.array(self.weights) - lr * grad).tolist()

        return {
            "loss": float(np.mean(error**2)),
            "weights": self.weights
        }
