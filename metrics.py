from abc import ABC
import torch
from typing import Any, List

class NLIModel(ABC):
    def __init__(self, name: str, hf_model_str: str) -> None:
        """
        Params:
            name: name of the model.
            hf_model_str: HuggingFace model string representing model weights on huggingface.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_str)
    
    @abstractmethod
    def entails(self, str1: str, str2: str) -> bool:
        """Find if str1 logically entails str2.

        Returns: True if str1 logicially entails str2.
        """
        pass
    
    def iff(self, str1: str, str2: str) -> bool:
        """Check if str1 entials str2 and vice versa.
        """
        return (entails(str1, str2) and entails(str2, str1))

class DebertaMNLIModel(NLIModel):
    def __init__(self):
        super().__init__("Deberta Large MNLI", "microsoft/deberta-large-mnli")
    
    def entails(self, str1, str2):
        encoded_input = tokenizer.encode(input, padding=True)
        prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
        predicted_label = torch.argmax(prediction, dim=1)
        return (predicted_label == 0)

class Metric(ABC):
    def __init__(self, name: str):
        """
        Params:
            name: The name of the metric.
        """
        self.name = name
    
    @abstractmethod
    def compute(self, generations: List[List[str]], loglikelihoods: torch.Tensor) -> torch.Tensor:
        """Computes the metric score on the inputs
        
        Params:
            generations: Outputs generated from prompts. Generations[i] corresponds to
                generations for the ith prompt.
                Len(generations) == batch_size, len(generations[i]) == outputs_per_prompt.
            loglikelihoods: Float tensor of shape (batch_size, outputs_per_prompt).
                loglikelihoods[i][j] is the likelihood of the jth sequence in batch i.
        """
        pass

class SemanticEntropy(Metric):
    def __init__(self, nli_model: NLIModel = DebertaMNLIModel()) -> None:
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
        equivalence_classes = [i for i in range(len(inputs))]
        for i in range(len(inputs)):
            for j in range(i, len(inputs)):
                if self.nli_model.iff(inputs[i], inputs[j]):
                    equivalence_classes[j] = equivalence_classes[i]
                    break
        
        return equivalence_classes
    
    def compute(self, generations, loglikelihoods):
        """Compute the Semantic Entropy.
        """
        equivalence_classes = [compute_equivalence_classes(gen) for gen in generations]
        equivalence_classes = torch.LongTensor(equivalence_classes)
        aggregated_likelihoods = torch_scatter.scatter_logsumexp(
            log_likelihoods, semantic_set_ids
        )
        return aggregated_likelihoods.mean(-1)
