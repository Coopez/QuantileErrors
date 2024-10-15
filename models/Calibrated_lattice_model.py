from layers.calibrated_lattice_layer import CalibratedLatticeLayer
import torch
import torch.nn as nn

from typing import Optional, Union
from pytorch_lattice.models.features import NumericalFeature, CategoricalFeature

from typing import Optional, Union

from pytorch_lattice.enums import (
    Interpolation,
    LatticeInit,
)
import warnings

from pytorch_lattice.utils.models import (
    calibrate_and_stack,
    initialize_feature_calibrators,
    initialize_monotonicities,
)
class CalibratedLatticeModel(nn.Module):
    """
    A PyTorch module that implements a calibrated lattice model with multiple layers.
    Args:
        features (list[Union[NumericalFeature, CategoricalFeature]]): List of features used in the model.
        clip_inputs (bool, optional): Whether to clip inputs to the lattice. Defaults to True.
        output_min (Optional[float], optional): Minimum value for the output. Defaults to None.
        output_max (Optional[float], optional): Maximum value for the output. Defaults to None.
        kernel_init (LatticeInit, optional): Initialization method for the lattice kernel. Defaults to LatticeInit.LINEAR.
        interpolation (Interpolation, optional): Interpolation method for the lattice. Defaults to Interpolation.HYPERCUBE.
        output_calibration_num_keypoints (Optional[int], optional): Number of keypoints for output calibration. Defaults to None.
        lattice_type (str, optional): Type of lattice to use. Defaults to 'lattice'.
        num_layers (int, optional): Number of layers in the model. Defaults to 1.
        output_size (int, optional): Size of the output. Defaults to 1.
    Attributes:
        lattice_layers (nn.ModuleList): List of lattice layers in the model.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the model.
    """

    def __init__(
        self,
        features: list[Union[NumericalFeature, CategoricalFeature]],
        clip_inputs: bool = True,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        num_layers: int = 1,
        input_dim_per_lattice: int = 1,
        num_lattice_first_layer: int = 1,
        output_size: int = 1,
        calibration_keypoints: int = 5,
    ) -> None:
        super().__init__()
        _decrease_factor = 2 # hardcoded for now - decreases the number of lattices in each layer
        self.features = features
        # grabbed from layers/calibrated_lattice_layer.py
        self.monotonicities = initialize_monotonicities(features)
        self.calibrators = initialize_feature_calibrators(
            features=features,
            output_min=0,
            output_max=[feature.lattice_size - 1 for feature in features],
        )
        self.input_dim_per_lattice = input_dim_per_lattice
        self.lattice_layers = []
        layer_size = num_lattice_first_layer

        # Construct lattice layers
        for i in range(num_layers-1):
            # need to stack layers
            lattice_layer = CalibratedLatticeLayer(
                monotonicities = self.monotonicities,
                clip_inputs=clip_inputs,
                output_min=output_min,
                output_max=output_max,
                kernel_init=kernel_init,
                interpolation=interpolation,
                input_dim_per_lattice = self.input_dim_per_lattice,
                num_lattice= layer_size,
                output_calibration_num_keypoints=calibration_keypoints,
            )
            layer_size = int(layer_size/_decrease_factor)
            
            self.input_dim_per_lattice = _decrease_factor * layer_size
            self.lattice_layers.append(lattice_layer)
            self.monotonicities = lattice_layer.lattice.output_monotonicities()

        # Last Lattice Layer
        self.lattice_layers.append(
            CalibratedLatticeLayer(
                monotonicities = self.monotonicities,
                clip_inputs=clip_inputs,
                output_min=output_min,
                output_max=output_max,
                kernel_init=kernel_init,
                interpolation=interpolation,
                input_dim_per_lattice = self.input_dim_per_lattice,
                num_lattice= layer_size,
                output_calibration_num_keypoints=calibration_keypoints,
                # output specific parameters
                output_size=output_size,
            )
        )
        self.lattice_layers = nn.ModuleList(self.lattice_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the calibrated lattice model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor after passing through all lattice layers.
        """
        x = calibrate_and_stack(x, self.calibrators)
        for lattice_layer in self.lattice_layers: #TODO: Does this need to be in a loop?
            x = lattice_layer(x)
        return x
