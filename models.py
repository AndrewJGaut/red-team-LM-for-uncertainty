from abc import ABC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""Abstract base class"""
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
        nli_input = f"{str1} [SEP] {str2}"
        encoded_input = tokenizer.encode(nli_input, padding=True)
        prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
        predicted_label = torch.argmax(prediction, dim=1)
        return (predicted_label == 0)
