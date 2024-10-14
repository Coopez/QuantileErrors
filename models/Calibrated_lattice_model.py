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
        output_calibration_num_keypoints: Optional[int] = None,
        lattice_type: str = 'lattice',
        num_layers: int = 1,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.lattice_layers = []
        out = int(len(features)/2)
        for i in range(num_layers-1):
            # need to stack layers
            if out > output_size:
                lattice_layer = CalibratedLatticeLayer(
                    features=features,
                    clip_inputs=clip_inputs,
                    output_min=output_min,
                    output_max=output_max,
                    kernel_init=kernel_init,
                    interpolation=interpolation,
                    output_calibration_num_keypoints=output_calibration_num_keypoints,
                    lattice_type=lattice_type,
                    output_size=int(len(features)/2),
                )
                out = int(out/2)
                self.lattice_layers.append(lattice_layer)
            else:
                break
        
        self.lattice_layers.append(
            CalibratedLatticeLayer(
                features=features,
                clip_inputs=clip_inputs,
                output_min=output_min,
                output_max=output_max,
                kernel_init=kernel_init,
                interpolation=interpolation,
                output_calibration_num_keypoints=output_calibration_num_keypoints,
                lattice_type=lattice_type,
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

        for lattice_layer in self.lattice_layers: #TODO: Does this need to be in a loop?
            x = lattice_layer(x)
        return x
