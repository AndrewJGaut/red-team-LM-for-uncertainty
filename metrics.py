from abc import ABC
import torch
import torch_scatter
from typing import Any, List

from models import *

"""Abstract Base Class"""
class Metric(ABC):
    def __init__(self, name: str):
        """
        Params:
            name: The name of the metric.
        """
        self.name = name
    
    @abstractmethod
    def compute(self, generations: List[List[str]], log_likelihoods: torch.Tensor) -> torch.Tensor:
        """Computes the metric score on the inputs
        
        Params:
            generations: Outputs generated from prompts. Generations[i] corresponds to
                generations for the ith prompt.
                Len(generations) == batch_size, len(generations[i]) == outputs_per_prompt.
            log_likelihoods: Float tensor of shape (batch_size, outputs_per_prompt).
                log_likelihoods[i][j] is the likelihood of the jth sequence in batch i.
        """
        pass


class SemanticEntropy(Metric):
    def __init__(self, nli_model: NLIModel) -> None:
        """
        Params:
            nli_model: The model to use to check logical entailment of strings.
        """
        super().__init__("Semantic Entropy")
        self.nli_model = nli_model

    def compute_equivalence_classes(self, inputs: List[str]) -> List[int]:
        """Partition inputs into semantic equivalence classes.

        Params:
            inputs: Inputs to partition
        
        Returns: A list with the same length as inputs. The int at the ith index is
        the integer representing the equivalence class assigned to the ith input.
        """
        equivalence_classes = torch.arange(len(inputs)) #[i for i in range(len(inputs))]
        for i in range(len(inputs)):
            for j in range(i, len(inputs)):
                if self.nli_model.iff(inputs[i], inputs[j]):
                    equivalence_classes[j] = equivalence_classes[i]
                    break

        output, inverse_indices = torch.unique(equivalence_classes, return_inverse=True)

        return inverse_indices
    
    def compute(self, generations, log_likelihoods):
        """Compute the Semantic Entropy.
        """
        entropy = torch.zeros(len(generations), device=log_likelihoods.device)
        for idx, gen in enumerate(generations):
            equivalence_classes = self.compute_equivalence_classes(gen)
            equivalence_classes = torch.tensor(equivalence_classes, device=log_likelihoods.device) # LongTensor(equivalence_classes)
            print(log_likelihoods.shape, equivalence_classes.shape, log_likelihoods[idx], equivalence_classes)
            aggregated_likelihoods = torch_scatter.scatter_logsumexp(
                log_likelihoods[idx], equivalence_classes
            )
            entropy[idx] = aggregated_likelihoods.mean()
        return entropy

