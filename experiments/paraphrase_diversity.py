"""
Determine a viable backtranslation sampling scheme that optimizes for positive diversity
in results, and minimizes time complexity.

"""
from time import time
from json import dump
import torch
from itertools import product
from tqdm import tqdm
from click import command, option
from nltk.tokenize import word_tokenize


class Experiment:
    def __init__(
        self,
        enforce_conditions_fn=None,
        **kwargs,
    ):
        """
        :param enforce_conditions_fn: boolean function that is executed prior to each hyperparameter
            combination in order to to see whether the given hyperparameter choices can work together.
            By default, allows all hyperparamter combinations

        For other parameter options, see `self.hyperparameters`

        """
        self.enforce_conditions_fn = enforce_conditions_fn

        for hyperparameter_name in self.hyperparameter_names:
            setattr(self, hyperparameter_name, kwargs.get(hyperparameter_name, None))

    @property
    def hyperparameters(self):
        return {
            hyperparameter_name: (
                value
                if isinstance(value, (list, set))
                else [value]
            )
            for hyperparameter_name in self.hyperparameter_names
            for value in [getattr(self, hyperparameter_name)]
            if value is not None
        }

    @property
    def hyperparameter_names(self):
        return [
            "beam",
            "diverse_beam_group",
            "sampling",
            "sampling_topk",
            "sampling_topp",
            "diversity_rate",
            "temperature"
        ]

    def sweep_hyperparameters(self):
        for hyperparameter_values in product(*self.hyperparameters.values()):
            kwargs = dict(zip(self.hyperparameters.keys(), hyperparameter_values))

            if self.enforce_conditions_fn and not self.enforce_conditions_fn(kwargs):
                continue

            yield kwargs


def run_decode(forward_model, backward_model, text, trials, **kwargs):
    """
    Our decoder passes the backtranslation **kwargs to the forward model, and uses
    a regular 5-beam search on the backward pass

    """
    # Encode the same text multiple times to act as multiple trials that can be
    # GPU accelerated
    tokenized_sentences = [forward_model.encode(text)] * trials

    trial_results = forward_model.generate(
        tokenized_sentences,
        **kwargs,
    )

    # Encoding
    encoding_outputs = [
        [
            forward_model.decode(result["tokens"])
            for result in trial_result
        ]
        for trial_result in trial_results
    ]

    # Decoding
    return [
        backward_model.sample(encoding_output)
        for encoding_output in encoding_outputs
    ]


def measure_results(original_text, results):
    """
    We use an inverse IOU score to determine how diverse the results set is, compared to the original seed text
    Favors sequences that generate absolute quantity and quality of diverse results

    """
    text_original_tokens = set([word.lower() for word in word_tokenize(original_text)])

    text_augmented_tokens = [
        set([word.lower() for word in word_tokenize(augmented)])
        for augmented in results
    ]

    return sum([
        1 - (len(text_augmented_token & text_original_tokens) / len(text_augmented_token | text_original_tokens))
        for text_augmented_token in text_augmented_tokens
    ])


@command()
@option("--text", type=str, default="The Canadian retail industry is undergoing massive transformations.")
@option("--use-gpu", type=bool, default=False)
@option("--trials", type=int, default=5)
def main(text, use_gpu, trials):
    experiments = [
        Experiment(
            beam=[1, 5, 10, 15],
        ),
        Experiment(
            beam=[1, 2, 5, 10],
            sampling=True,
            sampling_topk=[5, 10, 15, 20, 50],
            temperature=[0.2, 0.5, 1.0, 1.2, 1.5],
        ),
        Experiment(
            beam=[1, 2, 5, 10],
            sampling=True,
            sampling_topp=[0.2, 0.5, 0.8, 0.9],
            temperature=[0.2, 0.5, 1.0, 1.2, 1.5],
        ),
        Experiment(
            # Beam must be divisible by diverse_beam_groups
            beam=[5, 10, 15, 20, 25, 30],
            diverse_beam_group=[5, 10, 15],
            temperature=[0.2, 0.5, 1.0, 1.5],
            enforce_conditions_fn=lambda kwargs: (kwargs["beam"] / kwargs["diverse_beam_group"]).is_integer()
        ),
    ]

    forward_model = torch.hub.load(
        "pytorch/fairseq",
        "transformer.wmt19.en-de.single_model",
        tokenizer="moses",
        bpe="fastbpe",
    )
    backward_model = torch.hub.load(
        "pytorch/fairseq",
        "transformer.wmt19.de-en.single_model",
        tokenizer="moses",
        bpe="fastbpe",
    )

    if use_gpu:
        forward_model.cuda()
        backward_model.cuda()

    experiment_results = []

    for experiment in experiments:
        for hyperparameters in tqdm(list(experiment.sweep_hyperparameters())):
            start = time()
            trial_results = list(run_decode(forward_model, backward_model, text, trials, **hyperparameters))
            end = time()

            duration = end - start
            trial_scores = [
                measure_results(text, trial_result)
                for trial_result in trial_results
            ]

            experiment_results.append({
                "hyperparameters": hyperparameters,
                "results": [list(set(results)) for results in trial_results],
                "score": sum(trial_scores) / len(trial_scores),
                "duration": duration,
            })

    with open("results.json", "w") as file:
        dump(experiment_results, file)


if __name__ == "__main__":
    main()