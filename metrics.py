from abc import ABC
from collections import Counter
import re
import string
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
    
    def __str__(self):
        return self.name


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
        equivalence_classes = [i for i in range(len(inputs))]
        for i in range(len(inputs)):
            for j in range(i, len(inputs)):
                if self.nli_model.iff(inputs[i], inputs[j]):
                    equivalence_classes[j] = equivalence_classes[i]
                    break

        return equivalence_classes
    
    def compute(self, generations: List[List[str]], log_likelihoods: torch.Tensor) -> torch.Tensor:
        """Computes the semantic entropy score on the inputs.
        
        Params:
            generations: Outputs generated from prompts. Generations[i] corresponds to
                generations for the ith prompt.
                Len(generations) == batch_size, len(generations[i]) == outputs_per_prompt.
            log_likelihoods: Float tensor of shape (batch_size, outputs_per_prompt).
                log_likelihoods[i][j] is the likelihood of the jth sequence in batch i.
        """
        equivalence_classes = torch.tensor([
            self.compute_equivalence_classes(gen)
            for gen in generations
        ], device=log_likelihoods.device)

        aggregated_likelihoods = torch_scatter.scatter_logsumexp(
            log_likelihoods, equivalence_classes
        )

        # A masked mean operation
        mask = aggregated_likelihoods != -torch.inf
        masked = torch.where(mask, aggregated_likelihoods, 0)
        entropy = -masked.sum(-1) / mask.sum(-1)
        return entropy


"""Abstract Base Class"""
class QAMetric(Metric):
    def __init__(self, name: str):
        super().__init__(name)

    def get_tokens(self, s):
        """From the official SQuAD 2.0 eval script."""
        if not s:
            return []
        return self.normalize_answer(s).split()
    
    def normalize_answer(self, s):
        """Convert to lowercase and remove punctuation, articles and extra whitespace.
        
        From the official SQuAD 2.0 eval script
        """

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @abstractmethod
    def _compute(self, pred_answer: str, answer: str) -> float:
        """Compute score for predicted answer given real answer.

        Params:
            pred_answer: Predicted answer.
            answer: Ground truth answer.
        
        Returns: The score corresponding to the prediction on the ground truth answer.
        """
        pass

    def compute(self, pred_answer: str, answers: List[str]) -> Tuple[int, float]:
        """Compute score for predicted answer given real answers.

        Params:
            pred_answer: Predicted answer.
            answers: List of ground truth answers.
        
        Returns: A tuple. First element is index of ground truth answers corresponding
            to the max evaluation metric score. Second element is that score.
        """
        vals = [self._compute(pred_answer, answer) for answer in answers]
        max_idx = torch.argmax(torch.tensor(vals))
        return max_idx, vals[max_idx]


class F1(QAMetric):
    def __init__(self):
        super().__init__("F1")
    
    def _compute(self, pred_answer, answer):
        """From the official SQuAD 2.0 eval script.
        """
        pred_answer = self.get_tokens(pred_answer)
        answer = self.get_tokens(answer)
        common = Counter(pred_answer) & Counter(answer)
        num_same = sum(common.values())
        if len(pred_answer) == 0 or len(answer) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(pred_answer == answer)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_answer)
        recall = 1.0 * num_same / len(answer)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

class EM(QAMetric):
    def __init__(self):
        super().__init__("EM")
    
    def _compute(self, pred_answer, answer):
        """From the official SQuAD 2.0 eval script.
        """
        return int(self.normalize_answer(pred_answer) == self.normalize_answer(answer))
