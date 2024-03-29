from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
import itertools
import random
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchdata
from torchtext.datasets import SQuAD1
from tqdm import tqdm
from typing import Optional

from metrics import *
from models import *

def _all_subclasses_mapping(cls):
    def _all_subclasses(cls):
        return {cls}.union(s for c in cls.__subclasses__() for s in _all_subclasses(c))
    return {m.__name__: m for m in _all_subclasses(cls)}

def train(train_iter, full_model, semantic_entropy, num_to_generate, learning_rate, alpha, writer=None):
    """Run training.
    """
    def save(log_dir, model):
        current_date = datetime.now()
        torch.save(model.state_dict(), f'{log_dir}/{current_date.isoformat()}.pt')

    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, full_model.red_team.generator.model.parameters()), lr=learning_rate)
    if writer is None:
        writer = SummaryWriter()
    try:
        for i, instance in enumerate(tqdm(train_iter)):
            # Zero gradients
            optimizer.zero_grad()

            # Forward step
            context, real_question, answers, _ = instance
            question, sequences, lm_lls, red_lls, orig_lls = full_model(context, real_question, answers, num_to_generate)
            
            # Get loss.
            entropy = semantic_entropy.compute(sequences, lm_lls).mean()
            kl = torch.nn.functional.kl_div(red_lls, orig_lls, log_target=True)
            loss = -entropy + alpha * kl

            # Backpropagation and update step
            loss.backward()
            optimizer.step()

            # Logging.
            writer.add_text('train/GeneratedQuestion', question, i)
            writer.add_text('train/RealQuestion', real_question, i)
            writer.add_scalar('train/Entropy', entropy.item(), i)
            writer.add_scalar('train/KL', kl.item(), i)
            writer.add_scalar('train/Loss', loss.item(), i)
            writer.flush()
            if i % (len(train_iter) // 50) == 0:
                save(writer.log_dir, full_model.red_team.generator.model)
    finally:
        save(writer.log_dir, full_model.red_team.generator.model)

def test(test_iter, full_model, qa_metrics, red_team=True, writer=None):
    full_model.eval()
    if writer is None:
        writer = SummaryWriter()
    qa_metric_results = defaultdict(list)
    for i, instance in enumerate(tqdm(test_iter)):
        # Forward step
        context, real_question, answers, _ = instance
        question, pred_answers, _, _, _ = full_model(context, real_question, answers, 1, red_team)
        pred_answer = pred_answers[0][0]

        # Evalute and log metrics.
        for metric in qa_metrics:
            idx, val = metric.compute(pred_answer, answers)
            qa_metric_results[metric].append(val)
            writer.add_text(f'test-{red_team}/{metric}-Answer', answers[idx], i)

        # Logging.
        for qa_metric, vals in qa_metric_results.items():
            writer.add_histogram(f'test-{red_team}/{metric}', vals)
        writer.add_text(f'test-{red_team}/GeneratedQuestion', question, i)
        writer.add_text(f'test-{red_team}/RealQuestion', real_question, i)
        writer.add_text(f'test-{red_team}/PredictedAnswer', pred_answer, i)
        writer.flush()

def main(
    language_model: str,
    red_team_model: str,
    nli_model: str,
    path_to_red_team_model: Optional[str],
    alpha: float,
    learning_rate: float,
    semantic_entropy_m: int,
    num_train_instances: Optional[int],
    num_dev_instances: Optional[int],
    num_test_instances: Optional[int]
) -> None:
    """Train red team model to create prompts which produce uncertain outputs from language model.
    """
    # Parse language model classes.
    language_model_pt = HFLanguageModel(language_model, False, device=0, max_length=60, torch_dtype=torch.float16)
    red_team_model_pt = HFLanguageModel(red_team_model, device=0)
    if language_model == red_team_model:
        orig_model_pt = language_model_pt
    else:
        orig_model_pt = HFLanguageModel(red_team_model, False, 0)

    # Parse metric classes
    nli_model_pt = _all_subclasses_mapping(NLIModel)[nli_model]()
    semantic_entropy = SemanticEntropy(nli_model_pt)

    # Get data iters.
    train_iter = SQuAD1(split='train').header(num_train_instances) if num_train_instances else SQuAD1(split='train')
    test_iter = SQuAD1(split='train').header(num_test_instances) if num_test_instances else SQuAD1(split='train')

    # Train and evaluate.
    full_model = FullPipeline(language_model_pt, red_team_model_pt, orig_model_pt)

    # Set up SummaryWriter.
    if path_to_red_team_model:
        log_dir = os.path.dirname(path_to_red_team_model)
        writer = SummaryWriter(log_dir, filename_suffix='test')
    else:
        logdir = f"lr={learning_rate}_SEm={semantic_entropy_m}_alpha={alpha}_lm={language_model}_red-team={red_team_model}"
        writer = SummaryWriter(logdir)
    writer.add_hparams({
        'learning_rate': learning_rate,
        'semantic_entropy_m': semantic_entopy_m,
        'alpha': alpha,
        'language_model': language_model,
        'red_team_model': red_team_model,
        'nli_model': nli_model,
        'path_to_red_team_model': path_to_red_team_model
    })

    # Train and evaluate.
    if path_to_red_team_model:
        full_model.red_team.generator.model.load_state_dict(torch.load(path_to_red_team_model))
    else:
        train(train_iter, full_model, semantic_entropy, semantic_entropy_m, learning_rate, alpha, writer)
    test(test_iter, full_model, [F1(), EM()], True, writer)
    test(test_iter, full_model, [F1(), EM()], False, writer)

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
        default='vvsotnikov/stablelm-tuned-alpha-3b-16bit',
    )
    parser.add_argument(
        '-rt',
        '--red-team-model',
        type=str,
        help="Huggingface string for red team model",
        default='mrm8488/t5-base-finetuned-question-generation-ap',
    )
    parser.add_argument(
        '-nli',
        '--nli-model',
        type=str,
        help="String for NLI model. Possible values: [DebertaMNLIModel]",
        default='DebertaMNLIModel'
    )
    parser.add_argument(
        '--path-to-red-team-model',
        type=str,
        help="Path to saved red team model state dict. If not None, training will be skipped.",
        default=None
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        help="Learning rate of the model",
        default=1e-5
    )
    parser.add_argument(
        '-a',
        '--alpha',
        type=float,
        help="KL penalty factor",
        default=1e8#2.5e7
    )
    parser.add_argument(
        '--semantic-entropy-m',
        type=int,
        help="Number of generations to use for computing semantic entropy",
        default=4
    )
    parser.add_argument(
        '--train-dataset-size',
        type=int,
        help="Number of (context, answer) pairs to use for training",
        default=10000,
    )
    parser.add_argument(
        '--dev-dataset-size',
        type=int,
        help="Number of (context, answer) pairs to use for dev",
        default=1200
    )
    parser.add_argument(
        '--test-dataset-size',
        type=int,
        help="Number of (context, answer) pairs to use for test",
        default=1200
    )
    args = parser.parse_args()
    main(
        args.language_model,
        args.red_team_model,
        args.nli_model,
        args.path_to_red_team_model,
        args.alpha,
        args.learning_rate,
        args.semantic_entropy_m,
        args.train_dataset_size,
        args.dev_dataset_size,
        args.test_dataset_size
    )
