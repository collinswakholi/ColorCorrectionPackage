"""Load-only estimators that 2.x fitters stored inside saved calibrations.

``CustomNN`` (the ``legacy-v1`` ``custom`` CC method) and
``DeterministicTorchRegressor`` (the ``reproducible-v1`` ``custom`` method) are
kept so their pickles unpickle and predict exactly as before. Neither can be
trained any more. The pickled class paths are preserved:
``ColorCorrectionPipeline.core.correction.CustomNN`` and
``ColorCorrectionPipeline.fitting.DeterministicTorchRegressor`` resolve here.
Importing this module imports scikit-learn; load it only when a model needs it.
"""

import gc
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["CustomNN", "DeterministicTorchRegressor"]


class CustomNN(BaseEstimator, RegressorMixin):
    """Inference for the 2.x ``custom`` CC estimator (a PyTorch MLP).

    Load-only: 3.0 does not train it. Unpickled instances keep their fitted
    ``model`` (``torch.nn.Sequential``) and ``device``; ``predict`` is unchanged.
    """

    def __init__(
        self,
        hidden_layers=[64],
        optim_type="Adam",
        random_state=42,
        learning_rate=0.001,
        max_epochs=1000,
        tol=1e-6,
        verbose=False,
        batch_size=32,
        patience=10,
        dropout_rate=0.2,
        use_batch_norm=False,
    ):
        self.hidden_layers = hidden_layers
        self.optim_type = optim_type
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.verbose = verbose
        self.batch_size = batch_size
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.validation_split = 0.15
        self._requested_batch_norm = use_batch_norm

        self.input_dim: Any = None
        self.output_dim: Any = None
        self.temp_path: Optional[str] = None

        import torch.nn as nn

        self.model: Any = None  # nn.Sequential once built
        self.loss_fn = nn.MSELoss()
        self.device: Optional[str] = None

    def predict(self, X):
        """
        Predict using fitted neural network.

        Args:
            X: Input features (N, M)

        Returns:
            np.ndarray: Predictions (N, K)
        """
        self.model.eval()
        import torch

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            try:
                predictions = self.model(X_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Batch predictions
                    torch.cuda.empty_cache() if self.device == "cuda" else None
                    batch_size = 2
                    success = False
                    while not success:
                        try:
                            predictions = torch.cat(
                                [
                                    self.model(X_tensor[i : i + batch_size])
                                    for i in range(0, X_tensor.shape[0], batch_size)
                                ],
                                dim=0,
                            )
                            success = True
                        except RuntimeError as e2:
                            if "out of memory" in str(e2):
                                if batch_size == 1:
                                    if self.device == "cuda":
                                        cpu_model = self.model.to("cpu")
                                        cpu_tensor = X_tensor.detach().cpu()
                                        predictions = cpu_model(cpu_tensor)
                                        self.model = cpu_model
                                        self.device = "cpu"
                                        success = True
                                        continue
                                    raise RuntimeError(
                                        "CustomNN prediction ran out of memory even "
                                        "with batch_size=1."
                                    ) from e2
                                batch_size = max(1, batch_size // 2)
                                torch.cuda.empty_cache() if self.device == "cuda" else None
                            else:
                                raise e2
                else:
                    raise e

            results = predictions.cpu().numpy()

        torch.cuda.empty_cache() if self.device == "cuda" else None
        gc.collect()

        return results


# Pickles and repr use the historical location; core.correction resolves it lazily.
CustomNN.__module__ = "ColorCorrectionPipeline.core.correction"


class DeterministicTorchRegressor:
    """Inference for the ``reproducible-v1`` CPU NN; its fitter was removed in 3.0."""

    model: Any  # torch.nn.Sequential restored from the pickle

    def predict(self, X):
        import torch

        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor(np.asarray(X), dtype=torch.float32, device="cpu")
            ).numpy()


# Pickles reference the historical locations; those modules resolve these lazily.
DeterministicTorchRegressor.__module__ = "ColorCorrectionPipeline.fitting"
