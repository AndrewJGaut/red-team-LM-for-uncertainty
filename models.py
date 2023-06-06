from abc import ABC, abstractmethod
import random
import torch
from torch.nn import Module
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Tuple


class FullPipeline(Module):
    """Class that runs the full RedTeam-LanguageModel pipeline.
    """
    def __init__(self, lm, red_team, red_team_original):
        super().__init__()
        self.lm = lm
        self.red_team = red_team
        self.red_team_original = red_team_original
    
    def forward(self, context, question, answers, num_to_generate, use_red_team=True):
        """One forward pass over the whole model.
        """
        # Get prompt from context and answers.
        if use_red_team:
            question_generation_prompt_dec = self.red_team.generate_prompt(context, answers)
        else:
            question_generation_prompt_dec = self.red_team_original.generate_prompt(context, answers)

        # Generate questions and log-likelihoods with Red Team Model and feed into Language Model.
        prompts, red_lls, prompts_dec = self.red_team._generate_batch_for_prompt(question_generation_prompt_dec, 1, labels=question)
        with torch.no_grad():
            orig_lls = self.red_team_original.cond_probs(prompts, question)
            sequences, lm_lls = self.lm.generate_batch_for_prompts(prompts_dec, num_to_generate)
        return prompts_dec[0], sequences, lm_lls, red_lls, orig_lls



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
        prediction = self.model(encoded_input)['logits']
        predicted_label = torch.argmax(prediction, dim=1)
        return (predicted_label == 0)


class HFLanguageModel():
    def __init__(self, hf_model_str: str, return_full_text: bool = True, device: int = -1, max_length: int = 30, torch_dtype = None) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(hf_model_str)
        # self.model = AutoModelForCausalLM.from_pretrained(hf_model_str)
        self.generator = pipeline(
            'text-generation', model=hf_model_str, device=device, return_tensors=True, torch_dtype=torch_dtype) 
        self.max_length = max_length
        self.device = device

    def _generate_batch_for_prompt(self, prompt, num_to_generate, labels=None):
        sequences = self.generator(prompt, max_length=self.max_length, num_return_sequences=num_to_generate, num_beams=num_to_generate)
        sequences = torch.tensor([sequence['generated_token_ids'] for sequence in sequences], device=self.device)
        cond_probs = self.cond_probs(sequences, labels)
        decoded = self.decode(sequences)
        return sequences, cond_probs, decoded
        # TODO: Make torch Batch object
    
    def generate_batch_for_prompts(self, prompts, num_to_generate):
        sequences = []
        log_likelihoods = []
        for prompt in prompts:
            _, sequence_lls, sequence_dec = self._generate_batch_for_prompt(prompt, num_to_generate)
            sequences.append(sequence_dec)
            log_likelihood = sequence_lls.amax(-1).sum(-1)
            log_likelihoods.append(log_likelihood)
        log_likelihoods = torch.stack(log_likelihoods)
        return sequences, log_likelihoods
    
    def generate_prompt(self, context: str, answers: List[str]) -> Tuple[str, str]:
        """Generate prompt for question generation models.
        """
        answer = random.sample(answers, 1)  # Just take one answer.
        return f"answer: {answer}. context: {context}"



    def logits(self, sequences, labels=None):
        if labels:
            labels = self.generator.tokenizer(labels)
            labels = torch.tensor(labels['input_ids']).to(sequences.device)
            labels = labels.unsqueeze(0)
            return self.generator.model(sequences, labels=labels).logits
        return self.generator.model(sequences).logits

    def cond_probs(self, sequences, labels=None):
        return torch.nn.functional.log_softmax(self.logits(sequences, labels), -1)

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

