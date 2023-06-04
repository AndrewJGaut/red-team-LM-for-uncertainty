from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, pipeline

"""Abstract base class"""
class NLIModel(ABC):
    def __init__(self, name: str, hf_model_str: str) -> None:
        """
        Params:
            name: name of the model.
            hf_model_str: HuggingFace model string representing model weights on huggingface.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_str).cuda()

    @abstractmethod
    def entails(self, str1: str, str2: str) -> bool:
        """Find if str1 logically entails str2.

        Returns: True if str1 logicially entails str2.
        """
        pass

    def iff(self, str1: str, str2: str) -> bool:
        """Check if str1 entials str2 and vice versa.
        """
        return (self.entails(str1, str2) and self.entails(str2, str1))


class DebertaMNLIModel(NLIModel):
    def __init__(self):
        super().__init__("Deberta Large MNLI", "microsoft/deberta-large-mnli")

    def entails(self, str1, str2):
        nli_input = f"{str1} [SEP] {str2}"
        encoded_input = self.tokenizer.encode(nli_input, padding=True, return_tensors='pt').to(self.model.device)
        prediction = self.model(encoded_input.unsqueeze(0))['logits']
        predicted_label = torch.argmax(prediction, dim=1)
        return (predicted_label == 0)


class HFLanguageModel():
    def __init__(self, hf_model_str: str, return_full_text: bool = True, device: int = -1) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str)
        # self.model = AutoModelForCausalLM.from_pretrained(hf_model_str)
        self.generator = pipeline(
            'text-generation', model=hf_model_str, device=device, return_tensors=True, return_full_text=return_full_text
        ) # device_map="auto"
        self.max_length = 30
        self.device = device

    def generate_batch(self, prompt, num_to_generate):
        sequences = self.generator(prompt, max_length=self.max_length, num_return_sequences=num_to_generate)
        #print(prompt, sequences)
        sequences = torch.tensor([sequence['generated_token_ids'] for sequence in sequences], device=self.device)
        cond_probs = self.cond_probs(sequences)
        decoded = self.decode(sequences)
        return sequences, cond_probs, decoded
        # TODO: Make torch Batch object

        # prompt_enc = self.generator.tokenizer(
        #     prompt, padding=False, add_special_tokens=False, return_tensors='pt'
        # ).to(self.generator.model.device)
        # sequences = []
        # cond_probs = []
        # for _ in range(num_to_generate):
        #     tokens = self.generator.model.generate(prompt_enc, max_new_tokens=50, return_dict_in_generate=True, output_scores=True)
        #     sequences.append(tokens)
        #     cond_probs.append(self.cond_probs(tokens))
        # decoded = self.decode(sequences)
        # return sequences, cond_probs, decoded


    def logits(self, sequences):
        return self.generator.model(sequences).logits

    def cond_probs(self, sequences):
        return torch.nn.functional.log_softmax(self.logits(sequences), -1)

    def decode(self, sequences):
        #print('23s', sequences)
        return [
            self.generator.tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for sequence in sequences
        ]


'''
Some weights of the model checkpoint at microsoft/deberta-large-mnli were not used when initializing DebertaForSequenceClassification: ['config']
- This IS expected if you are initializing DebertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

'''
#TensorDict({'sequences': })

