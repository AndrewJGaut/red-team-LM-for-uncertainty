from argparse import ArgumentParser
import copy
from datetime import datetime
import itertools

from metrics import *
from models import *

import torch
from torch.utils.tensorboard import SummaryWriter


def save(log_dir, model):
    current_date = datetime.now()
    torch.save(model, f'{log_dir}/{current_date.isoformat()}.pt')

def main(
    language_model: str,
    red_team_model: str,
    nli_model: str,
    alpha: float,
    learning_rate: float
) -> None:
    """Train red team model to create prompts which produce uncertain outputs from language model.
    """
    # Parse model classes.
    language_model_pt = HFLanguageModel(language_model, 0)
    language_model_pt.max_length = 60
    red_team_model_pt = HFLanguageModel(red_team_model, 0)
    nli_model_pt = globals()[nli_model]()
    semantic_entropy = SemanticEntropy(nli_model_pt)

    if language_model == red_team_model:
        orig_model_pt = language_model_pt
    else:
        orig_model_pt = copy.deepcopy(red_team_model_pt)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, language_model_pt.generator.model.parameters()), lr=learning_rate)
    writer = SummaryWriter()

    try:
        for i in itertools.count():

            # Zero gradients
            optimizer.zero_grad()

            # Forward step
            prompts, red_lls, prompt_dec = red_team_model_pt.generate_batch("", 4)
            orig_lls = orig_model_pt.cond_probs(prompts)

            sequences = []
            log_likelihoods = []
            for prompt in prompt_dec:
                sequence_tokens, sequence_lls, sequence_dec = language_model_pt.generate_batch(prompt, 10)
                sequences.append(sequence_dec)
                log_likelihood = sequence_lls.amax(-1).sum(-1)
                log_likelihoods.append(log_likelihood)
            log_likelihoods = torch.stack(log_likelihoods)
            
            entropy = semantic_entropy.compute(sequences, log_likelihoods).sum()
            kl = torch.nn.functional.kl_div(red_lls, orig_lls, log_target=True)
            # (orig_logits - red_logits).sum(-1) # TODO: Figure this out later
            loss = -entropy + alpha * kl

            # Backpropagation and update step
            loss.backward()
            optimizer.step()

            writer.add_scalar('Entropy', entropy.item(), i)
            writer.add_scalar('KL', kl.item(), i)
            writer.add_scalar('Loss', loss.item(), i)
            writer.flush()

            if i % 500 == 499:
                save(writer.log_dir, red_team_model_pt)
    finally:
        save(writer.log_dir, red_team_model_pt)


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='red_team.py',
        description='Trains a red team language model using RL to product prompts which elicit generations from another language model for which that other model is very uncertain'
    )  
    parser.add_argument(
        '-lm',
        '--language-model',
        type=str,
        help="Huggingface string for model that is being adversarially attacked by the red team model",
        default='gpt2',#'stabilityai/stablelm-tuned-alpha-3b'
    )
    parser.add_argument(
        '-rt',
        '--red-team-model',
        type=str,
        help="Huggingface string for red team model",
        default='gpt2',#'stabilityai/stablelm-tuned-alpha-3b'
    )
    parser.add_argument(
        '-nli',
        '--nli-model',
        type=str,
        help="String for red team model. Possible values: [DebertaMNLIModel]",
        default='DebertaMNLIModel'
    )
    parser.add_argument(
        '-nli',
        '--nli-model',
        type=str,
        help="String for red team model. Possible values: [DebertaMNLIModel]",
        default='DebertaMNLIModel'
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        help="Learning rate of the model",
        default=1e-5
    )
    parser.add_argument(
        '-alpha',
        '--alpha',
        type=float,
        help="KL penalty factor",
        default=1e7
    )
    # Todo... add more arguments.
    args = parser.parse_args()
    main(
        args.language_model,
        args.red_team_model,
        args.nli_model,
        args.alpha,
        args.learning_rate
    )



