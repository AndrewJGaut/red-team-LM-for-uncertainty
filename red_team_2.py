
from transformers import Trainer, TrainingArguments
from models import *
from metrics import *

from dataclasses import dataclass, field
import itertools

@dataclass
class RedTeamTrainingArguments(TrainingArguments):
    alpha: float = field(
        default=2.5e6,
        metadata={
            "help": (
                "The KL penalty weight."
            )
        },
    )

class RedTeamTrainer(Trainer):
    args: RedTeamTrainingArguments
    orig_model_pt: HFLanguageModel
    language_model_pt: HFLanguageModel
    semantic_entropy: Metric

    def get_train_dataloader(self): return itertools.repeat({})
    def get_eval_dataloader(self): return itertools.repeat({})
    def get_test_dataloader(self): return itertools.repeat({})

    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = 4
        gen_count = 10

        red_team_model_pt = model
        
        prompts, red_lls, prompt_dec = red_team_model_pt.generate_batch("", batch_size)
        orig_lls = self.orig_model_pt.cond_probs(prompts)

        sequences = []
        log_likelihoods = []
        for prompt in prompt_dec:
            _, sequence_lls, sequence_dec = self.language_model_pt.generate_batch(prompt, gen_count)
            sequences.append(sequence_dec)
            log_likelihood = sequence_lls.amax(-1).sum(-1)
            log_likelihoods.append(log_likelihood)
        log_likelihoods = torch.stack(log_likelihoods)
        
        entropy = self.semantic_entropy.compute(sequences, log_likelihoods).mean()
        kl = torch.nn.functional.kl_div(red_lls, orig_lls, log_target=True)
        # (orig_logits - red_logits).sum(-1) # TODO: Figure this out later
        loss = -entropy + self.args.alpha * kl
        return (loss, red_lls) if return_outputs else loss

