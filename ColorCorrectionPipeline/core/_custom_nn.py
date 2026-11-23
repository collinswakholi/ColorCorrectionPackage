"""PyTorch neural-network regressor used by the legacy ``custom`` CC method.

Kept separate so importing :mod:`ColorCorrectionPipeline.core.correction` does not import
scikit-learn (about 1.3 s). The class keeps its public identity
``ColorCorrectionPipeline.core.correction.CustomNN``, so existing pickles load unchanged and
new pickles reference the same name.
"""

import gc
import os
import tempfile
from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y

__all__ = ["CustomNN"]


class CustomNN(BaseEstimator, RegressorMixin):
    """
    Custom neural network regressor with PyTorch backend.

    Implements sklearn-compatible interface for color correction using
    fully connected neural networks with optional batch normalization and dropout.

    Args:
        hidden_layers: List of hidden layer sizes
        optim_type: Optimizer type ("Adam", "SGD", "RMSprop")
        random_state: Random seed for reproducibility
        learning_rate: Learning rate
        max_epochs: Maximum training epochs
        tol: Convergence tolerance
        verbose: Whether to print training progress
        batch_size: Mini-batch size
        patience: Early stopping patience
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization

    Example:
        >>> model = CustomNN(hidden_layers=[32, 16], max_epochs=100)
        >>> X_train = np.random.rand(100, 10)
        >>> y_train = np.random.rand(100, 3)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_train)
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

        self.input_dim = None
        self.output_dim = None
        self.temp_path: Optional[str] = None

        import torch.nn as nn

        self.model: Optional[nn.Module] = None
        self.loss_fn = nn.MSELoss()
        self.device: Optional[str] = None

    def _build_model(self):
        """Build neural network architecture."""
        import torch
        import torch.nn as nn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.verbose:
            print(f"Using device: '{self.device}'")

        layers = []
        current_dim = self.input_dim

        # Add hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim, eps=self.tol))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, self.output_dim))

        self.model = nn.Sequential(*layers).to(self.device)

        if self.verbose:
            print("Model architecture:")
            print(self.model)

    def fit(self, X, y):
        """
        Fit neural network to training data.

        ``random_state`` seeds weight initialisation, batch shuffling, dropout and
        the train/validation split, so equal seeds give equal models on the same
        device. The caller's global torch RNG state is restored afterwards.

        Args:
            X: Training features (N, M)
            y: Training targets (N, K)

        Returns:
            self: Fitted model
        """
        import torch

        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.random_state))
            try:
                return self._fit_seeded(X, y)
            finally:
                if self.temp_path and os.path.exists(self.temp_path):
                    os.remove(self.temp_path)
                self.temp_path = None

    def _fit_seeded(self, X, y):
        """Training body of :meth:`fit`; runs inside a seeded, forked RNG."""
        X, y = check_X_y(X, y, multi_output=True)
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1] if len(y.shape) > 1 else 1

        # Train/val split sizes. Compute these before model construction so
        # batch-normalization can be disabled for splits that are too small.
        dataset_size = len(X)
        if dataset_size < 2:
            train_size = dataset_size
            val_size = dataset_size
        else:
            val_size = max(1, int(dataset_size * self.validation_split))
            train_size = dataset_size - val_size

        self.use_batch_norm = self._requested_batch_norm
        if self.use_batch_norm and (train_size < 2 or self.batch_size < 2):
            if self.verbose:
                print(
                    "Disabling batch normalization because the training split "
                    "or batch size is too small."
                )
            self.use_batch_norm = False

        self._build_model()

        # Convert to tensors after batch-norm validation so model construction uses
        # the effective normalization setting.
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset, random_split

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )

        if dataset_size < 2:
            train_dataset = dataset
            val_dataset = dataset
        else:
            generator = torch.Generator().manual_seed(self.random_state)
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )

        with tempfile.NamedTemporaryFile(
            prefix="cc_best_model_", suffix=".pth", delete=False
        ) as temp_file:
            self.temp_path = temp_file.name

        pin_memory = self.device == "cuda"

        drop_last_train_batch = (
            self.use_batch_norm
            and len(train_dataset) > self.batch_size
            and len(train_dataset) % self.batch_size == 1
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(self.random_state)),
            pin_memory=pin_memory,
            drop_last=drop_last_train_batch,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=pin_memory
        )

        # Optimizer
        if self.optim_type == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optim_type == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optim_type == "RMSprop":
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.patience // 2
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        n_verbose = 50
        interval = max(1, int(self.max_epochs / n_verbose))

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_X)
                    loss = self.loss_fn(predictions, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss - self.tol:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                torch.save(self.model.state_dict(), self.temp_path)
            else:
                epochs_no_improve += 1

            if self.verbose and (epoch % interval == 0 or epoch == self.max_epochs - 1):
                print(
                    f"Epoch {epoch+1}/{self.max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model. The checkpoint file is empty if validation never
        # improved (e.g. non-finite loss); keep the last weights in that case.
        if self.temp_path and os.path.exists(self.temp_path) and os.path.getsize(self.temp_path):
            self.model.load_state_dict(torch.load(self.temp_path, map_location=self.device))
            os.remove(self.temp_path)
            self.temp_path = None

        # Release CUDA resources
        if self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

        return self

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
