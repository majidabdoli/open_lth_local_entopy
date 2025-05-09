# Copyright (c) Piota, majidabdoli
# window of pruning change by WinSize

import dataclasses
import numpy as np

from scipy.stats import entropy
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
        WinSize=8
        wightsentropy=[]
        for k, v in weights.items():
            for i in range (0,v.shape[0]-WinSize,WinSize):
                for j in range (0,v.shape[1]-WinSize,WinSize):
                    weight_vector=v[i:i+WinSize,j:j+WinSize].flatten()
                    weight_vector=weight_vector[weight_vector!=0]
                    wightsentropy.append(entropy(np.abs(weight_vector)))
        wightEntopyMedian=np.median(wightsentropy)
        sparseEntropyDiff=20
        for k, v in weights.items():
            for i in range (0,v.shape[0]-WinSize,WinSize):
                for j in range (0,v.shape[1]-WinSize,WinSize):
                    weight_vector=v[i:i+WinSize,j:j+WinSize].flatten()
                    weight_vector=weight_vector[weight_vector!=0]
                    # Determine the number of weights that need to be pruned.
                    local_mask=current_mask[k][i:i+WinSize,j:j+WinSize]
                    number_of_remaining_weights = np.sum([np.sum(vl) for vl in local_mask])
                    number_of_weights_to_prune = np.ceil(
                        pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)
                    if(entropy(np.abs(weight_vector))<wightEntopyMedian):
                        number_of_weights_to_prune+=int(number_of_weights_to_prune/8)
                    else:
                        number_of_weights_to_prune-=int(number_of_weights_to_prune/8)
                    #weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
                    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]
                    new_mask_local = Mask({k: np.where(np.abs(v[i:i+WinSize,j:j+WinSize]) > threshold,
                                                        local_mask,
                                                        np.zeros_like(v[i:i+WinSize,j:j+WinSize]))
                                    })
                    current_mask[k][i:i+WinSize,j:j+WinSize]=new_mask_local[k]
        for k in current_mask:
            if k not in current_mask_n:
                current_mask_n[k] = current_mask[k]
        
        print('Lottary with Win{}'.format(WinSize))
        
        return current_mask_n
