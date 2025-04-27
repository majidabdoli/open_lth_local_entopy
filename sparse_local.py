# Copyright (c) Piota, majidabdoli

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask_n = Mask.ones_like(trained_model) if current_mask is None else current_mask
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
        # Determine the number of weights that need to be pruned.
        # number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        # number_of_weights_to_prune = np.ceil(
        #     pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}
        # Create a vector of all the unpruned weights in the model.
        #weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        winSize=8
        for k, v in weights.items():
            for i in range (0,v.shape[0]-winSize,winSize):
                for j in range (0,v.shape[1]-winSize,winSize):
                     # Determine the number of weights that need to be pruned.
                    local_mask=current_mask[k][i:i+winSize,j:j+winSize]
                    number_of_remaining_weights = np.sum([np.sum(vl) for vl in local_mask])
                    number_of_weights_to_prune = np.ceil(
                        pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)
                    weight_vector=v[i:i+winSize,j:j+winSize].flatten()
                    weight_vector=weight_vector[weight_vector!=0]
                    #weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
                    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]
                    new_mask_local = Mask({k: np.where(np.abs(v[i:i+winSize,j:j+winSize]) > threshold,
                                                        local_mask,
                                                        np.zeros_like(v[i:i+winSize,j:j+winSize]))
                                    })
                    current_mask[k][i:i+winSize,j:j+winSize]=new_mask_local[k]
        for k in current_mask:
            if k not in current_mask_n:
                current_mask_n[k] = current_mask[k]

        return current_mask_n
