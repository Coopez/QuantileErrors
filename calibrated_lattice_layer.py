"""Class for easily constructing a calibrated lattice model."""
from typing import Optional, Union

import torch

from pytorch_lattice.constrained_module import ConstrainedModule
from pytorch_lattice.enums import (
    Interpolation,
    LatticeInit,
    Monotonicity,
)
from pytorch_lattice.layers import Lattice
from pytorch_lattice.utils.models import (
    calibrate_and_stack,
    initialize_feature_calibrators,
    initialize_monotonicities,
    initialize_output_calibrator,
)
from pytorch_lattice.models.features import CategoricalFeature, NumericalFeature
from pytorch_lattice.layers import RTL


import numpy as np
import pandas as pd

class CalibratedLatticeLayer(ConstrainedModule):
    """PyTorch Calibrated Lattice Model.

    Creates a `torch.nn.Module` representing a calibrated lattice model, which will be
    constructed using the provided model configuration. Note that the model inputs
    should match the order in which they are defined in the `feature_configs`.

    Attributes:
        All: `__init__` arguments.
        calibrators: A dictionary that maps feature names to their calibrators.
        lattice: The `Lattice` layer of the model.
        output_calibrator: The output `NumericalCalibrator` calibration layer. This
            will be `None` if no output calibration is desired.

    Example:

    ```python
    feature_configs = [...]
    calibrated_model = CalibratedLattice(feature_configs, ...)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_model.parameters(recurse=True), lr=1e-1)

    dataset = pyl.utils.data.Dataset(...)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(100):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = calibrated_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            calibrated_model.apply_constraints()
    ```
    """

    def __init__(
        self,
        features: list[Union[NumericalFeature, CategoricalFeature]],
        clip_inputs: bool = True,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        output_calibration_num_keypoints: Optional[int] = None,
        lattice_type: str = 'lattice',
        output_size: int = 1,
    ) -> None:
        """Initializes an instance of `CalibratedLattice`.

        Args:
            features: A list of numerical and/or categorical feature configs.
            clip_inputs: Whether to restrict inputs to the bounds of lattice.
            output_min: The minimum output value for the model. If `None`, the minimum
                output value will be unbounded.
            output_max: The maximum output value for the model. If `None`, the maximum
                output value will be unbounded.
            kernel_init: the method of initializing kernel weights. If otherwise
                unspecified, will default to `LatticeInit.LINEAR`.
            interpolation: the method of interpolation in the lattice's forward pass.
                If otherwise unspecified, will default to `Interpolation.HYPERCUBE`.
            output_calibration_num_keypoints: The number of keypoints to use for the
                output calibrator. If `None`, no output calibration will be used.

        Raises:
            ValueError: If any feature configs are not `NUMERICAL` or `CATEGORICAL`.
        """
        super().__init__()

        self.features = features
        self.clip_inputs = clip_inputs
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.interpolation = interpolation
        self.lattice_type = lattice_type
        self.output_calibration_num_keypoints = output_calibration_num_keypoints
        self.monotonicities = initialize_monotonicities(features)
        self.calibrators = initialize_feature_calibrators(
            features=features,
            output_min=0,
            output_max=[feature.lattice_size - 1 for feature in features],
        )
        self.output_size = output_size
        if lattice_type == 'lattice':
            self.lattice = Lattice(
                lattice_sizes=[feature.lattice_size for feature in features],
                monotonicities=self.monotonicities,
                clip_inputs=self.clip_inputs,
                output_min=self.output_min,
                output_max=self.output_max,
                interpolation=interpolation,
                kernel_init=kernel_init,
            )
        elif lattice_type == 'rtl':
            self.lattice = RTL(
                num_lattices= len(features),
                lattice_rank=1,
                monotonicities=self.monotonicities,
                clip_inputs=self.clip_inputs,
                output_min=self.output_min,
                output_max=self.output_max,
                interpolation=interpolation,
                kernel_init=kernel_init,
            )
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
        # self.lattice = Lattice(
        #     lattice_sizes=[feature.lattice_size for feature in features],
        #     monotonicities=self.monotonicities,
        #     clip_inputs=self.clip_inputs,
        #     output_min=self.output_min,
        #     output_max=self.output_max,
        #     interpolation=interpolation,
        #     kernel_init=kernel_init,
        # )

        self.output_calibrator = initialize_output_calibrator(
            output_calibration_num_keypoints=output_calibration_num_keypoints,
            monotonic=not all(m is None for m in self.monotonicities),
            output_min=output_min,
            output_max=output_max,
        )

        self.layer_output = Linear(len(features), self.output_size,self.monotonicities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs an input through the network to produce a calibrated lattice output.

        Args:
            x: The input tensor of feature values of shape `(batch_size, num_features)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing the model output result.
        """
        result = calibrate_and_stack(x, self.calibrators)
        result = self.lattice(result)
        if self.output_calibrator is not None:
            result = self.output_calibrator(result)
        result = self.layer_output(result)
        return result

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Constrains the model into desired constraints specified by the config."""
        for calibrator in self.calibrators.values():
            calibrator.apply_constraints()
        self.lattice.apply_constraints()
        if self.output_calibrator:
            self.output_calibrator.apply_constraints()

    @torch.no_grad()
    def assert_constraints(self, eps: float = 1e-6) -> dict[str, list[str]]:
        """Asserts all layers within model satisfied specified constraints.

        Asserts monotonicity pairs and output bounds for categorical calibrators,
        monotonicity and output bounds for numerical calibrators, and monotonicity and
        weights summing to 1 if weighted_average for linear layer.

        Args:
            eps: the margin of error allowed

        Returns:
            A dict where key is feature_name for calibrators and 'linear' for the linear
            layer, and value is the error messages for each layer. Layers with no error
            messages are not present in the dictionary.
        """
        messages = {}

        for name, calibrator in self.calibrators.items():
            calibrator_messages = calibrator.assert_constraints(eps)
            if calibrator_messages:
                messages[f"{name}_calibrator"] = calibrator_messages
        lattice_messages = self.lattice.assert_constraints(eps)
        if lattice_messages:
            messages["lattice"] = lattice_messages
        if self.output_calibrator:
            output_calibrator_messages = self.output_calibrator.assert_constraints(eps)
            if output_calibrator_messages:
                messages["output_calibrator"] = output_calibrator_messages

        return messages
    



class CalibratedDataset(torch.utils.data.Dataset):
    """A class for loading a dataset for a calibrated model."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: list[Union[NumericalFeature, CategoricalFeature]],
        window_size: int,
        horizon_size: int,
    ):
        self.horizon_size = horizon_size   
        self.window_size = window_size
        
        """Initializes an instance of `Dataset`."""
        self.X = X.copy()
        self.y = y.copy()
        
        selected_features = [feature.feature_name for feature in features]
        unavailable_features = set(selected_features) - set(self.X.columns)
        if len(unavailable_features) > 0:
            raise ValueError(f"Features {unavailable_features} not found in dataset.")

        drop_features = list(set(self.X.columns) - set(selected_features))
        self.X.drop(drop_features, inplace=True)
        self.quantiles = self.X.pop("quantiles") if "quantiles" in self.X.columns else None

        self.data = torch.from_numpy(self.X.values).double() #.to(torch.float32)
        self.targets = torch.from_numpy(self.y)[:, None].double()#.to(torch.float32)

    def __len__(self):
        return len(self.X) - self.window_size - self.horizon_size + 1

    def __getitem__(self, idx):
        # if isinstance(idx, torch.Tensor):
        #     idx = idx.tolist()
        x = self.data[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size : idx + self.window_size + self.horizon_size]
        if self.quantiles is not None:
            q = self.quantiles[idx + self.window_size].repeat(self.window_size)
            q = torch.from_numpy(q)[:, None].double()#.to(torch.float32)
            return [x, q, y]
        return [x, y]
    


    """Linear module for use in calibrated modeling.

PyTorch implementation of the calibrated linear module. This module takes in a
single-dimensional input and transforms it using a linear transformation and optionally
a bias term. This module supports monotonicity constraints.
"""

class Linear(ConstrainedModule):
    """A constrained linear module.

    This module takes an input of shape `(batch_size, input_dim)` and applied a linear
    transformation. 

    Attributes:
        All: `__init__` arguments.
        kernel: `torch.nn.Parameter` that stores the linear combination weighting.
        bias: `torch.nn.Parameter` that stores the bias term. Only available is
            `use_bias` is true.

    Example:
    ```python
    input_dim = 3
    inputs = torch.tensor(...)  # shape: (batch_size, input_dim)
    linear = Linear(
        input_dim,
        monotonicities=[
            None,
            Monotonicity.INCREASING,
            Monotonicity.DECREASING
        ],
        use_bias=False,
        weighted_average=True,
    )
    outputs = linear(inputs)
    ```
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        monotonicities: Optional[list[Optional[Monotonicity]]] = None,
        use_bias: bool = True,
        weighted_average: bool = False,
    ) -> None:
        """Initializes an instance of `Linear`.

        Args:
            input_dim: The number of inputs that will be combined.
            monotonicities: If provided, specifies the monotonicity of each input
                dimension.
            use_bias: Whether to use a bias term for the linear combination.
            weighted_average: Whether to make the output a weighted average i.e. all
                coefficients are positive and add up to a total of 1.0. No bias term
                will be used, and `use_bias` will be set to false regardless of the
                original value. `monotonicities` will also be set to increasing for all
                input dimensions to ensure that all coefficients are positive.

        Raises:
            ValueError: If monotonicities does not have length input_dim (if provided).
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        if monotonicities and len(monotonicities) != input_dim:
            raise ValueError("Monotonicities, if provided, must have length input_dim.")
        self.monotonicities = (
            monotonicities
            if not weighted_average
            else [Monotonicity.INCREASING] * input_dim
        )
        self.use_bias = use_bias if not weighted_average else False
        self.weighted_average = weighted_average

        self.kernel = torch.nn.Parameter(torch.Tensor(input_dim, output_dim).double())
        torch.nn.init.constant_(self.kernel, 1.0 / input_dim)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1).double())
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms inputs using a linear combination.

        Args:
            x: The input tensor of shape `(batch_size, input_dim)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing transformed input values.
        """
        result = torch.mm(x, self.kernel)
        if self.use_bias:
            result += self.bias
        return result

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Projects kernel into desired constraints."""
        projected_kernel_data = self.kernel.data

        if self.monotonicities:
            if Monotonicity.INCREASING in self.monotonicities:
                increasing_mask = torch.tensor(
                    [
                        [0.0] if m == Monotonicity.INCREASING else [1.0]
                        for m in self.monotonicities
                    ]
                )
                projected_kernel_data = torch.maximum(
                    projected_kernel_data, projected_kernel_data * increasing_mask
                )
            if Monotonicity.DECREASING in self.monotonicities:
                decreasing_mask = torch.tensor(
                    [
                        [0.0] if m == Monotonicity.DECREASING else [1.0]
                        for m in self.monotonicities
                    ]
                )
                projected_kernel_data = torch.minimum(
                    projected_kernel_data, projected_kernel_data * decreasing_mask
                )

        if self.weighted_average:
            norm = torch.norm(projected_kernel_data, 1)
            norm = torch.where(norm < 1e-8, 1.0, norm)
            projected_kernel_data /= norm

        self.kernel.data = projected_kernel_data

    @torch.no_grad()
    def assert_constraints(self, eps: float = 1e-6) -> list[str]:
        """Asserts that layer satisfies specified constraints.

        This checks that decreasing monotonicity corresponds to negative weights,
        increasing monotonicity corresponds to positive weights, and weights sum to 1
        for weighted_average=True.

        Args:
            eps: the margin of error allowed

        Returns:
            A list of messages describing violated constraints. If no constraints
            violated, the list will be empty.
        """
        messages = []

        if self.weighted_average:
            total_weight = torch.sum(self.kernel.data)
            if torch.abs(total_weight - 1.0) > eps:
                messages.append("Weights do not sum to 1.")

        if self.monotonicities:
            monotonicities_constant = torch.tensor(
                [
                    1
                    if m == Monotonicity.INCREASING
                    else -1
                    if m == Monotonicity.DECREASING
                    else 0
                    for m in self.monotonicities
                ],
                device=self.kernel.device,
                dtype=self.kernel.dtype,
            ).view(-1, 1)

            violated_monotonicities = (self.kernel * monotonicities_constant) < -eps
            violation_indices = torch.where(violated_monotonicities)
            if violation_indices[0].numel() > 0:
                messages.append(
                    f"Monotonicity violated at: {violation_indices[0].tolist()}"
                )

        return messages
