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
    language_model = HFLanguageModel(language_model)#.cuda()
    red_team_model = HFLanguageModel(red_team_model)
    nli_model = globals()[nli_model]

    # TODO...



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


