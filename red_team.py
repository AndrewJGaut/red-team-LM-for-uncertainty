from argparse import ArgumentParser

from metrics import *
from models import *


def main(
    language_model: str,
    red_team_model: str,
    nli_model: str
) -> None:
    """Train red team model to create prompts which produce uncertain outputs from language model.
    """
    # Parse model classes.
    language_model_pt = HFLanguageModel(language_model, 0)
    red_team_model_pt = HFLanguageModel(red_team_model, 0)
    nli_model_pt = globals()[nli_model]()
    semantic_entropy = SemanticEntropy(nli_model_pt)

    # TODO...
    prompts = red_team_model_pt.generate_batch("", 4)
    orig_logits = language_model_pt.logits(prompts)
    red_logits = red_team_model_pt.logits(prompts)

    sequences = []
    log_likelihoods = []
    for idx, prompt in enumerate(prompts):
        sequence_tokens = language_model_pt.generate_batch(prompt, 3)
        sequences.append(language_model_pt.decode(sequence_tokens))
        sequence_logits = language_model_pt.logits(sequence_tokens)
        log_likelihoods.append(sequence_logits.sum(-1))
    log_likelihoods = torch.stack(log_likelihoods)
    
    entropy = semantic_entropy.compute(sequences, log_likelihoods)
    kl = (orig_logits - red_logits).sum() # TODO: Figure this out later
    print(-entropy + kl)



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
    # Todo... add more arguments.
    args = parser.parse_args()
    main(
        args.language_model,
        args.red_team_model,
        args.nli_model
    )


