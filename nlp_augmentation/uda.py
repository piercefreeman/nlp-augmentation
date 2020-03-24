from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
from zipfile import ZipFile

from click import group, option, Path as ClickPath, Choice
from tqdm import tqdm
from json import loads as json_loads, dumps as json_dumps
from random import choices
from itertools import chain, groupby

from nlp_augmentation.backtranslation.backtranslate import BackTranslate
from nlp_augmentation.word_substitution.tfidf import TfIdfWordSubstitution
from nlp_augmentation.data_models import Datapoint


@group()
def uda():
    pass


@uda.command()
@option("--input-path", type=ClickPath(exists=True, file_okay=True), required=True)
@option("--augmentation-count", type=int, required=True)
@option("--output-path", type=ClickPath(), required=True)
@option(
    "--method",
    type=Choice([
        "cpu",
        "gpu",
    ]),
    required=True,
    default="cpu",
)
@option("--gpu-count", type=int, default=None)
@option("--workers", type=int, default=1)
def augment(input_path, augmentation_count, output_path, method, gpu_count, workers):
    """
    CLI utility to augment the text contents of an input file.  Expects a `.jsonl` file with
    each line containing a `text` key with the text that should be supplemented.

    Response format is a jsonl file where each line correlates to the original line.

    :param input-path: Path to the input `.jsonl` file.
    :param augmentation-count: Number of augmented datapoints to generate for each input datapoint.
    :param input-path: Path to the output location to write the augmented data.

    """
    examples = []
    with open(input_path) as file:
        for line in file:
            examples.append(json_loads(line)["text"])

    # We generate a consistent number of examples using each augmentation technique
    # Sample our scheme here
    augmentation_techniques = ["backtranslation", "tfidf"]
    augmentation_schemes = choices(augmentation_techniques, k=augmentation_count)

    scheme_count = {
        scheme: augmentation_schemes.count(scheme)
        for scheme in augmentation_techniques
    }

    backtranslation = BackTranslate(
        augmentations=scheme_count["backtranslation"],
        workers=workers,
        use_gpu=(method == "gpu"),
        gpu_count=gpu_count,
    )

    word_replacement = TfIdfWordSubstitution(
        augmentations=scheme_count["tfidf"],
        token_prob=0.7
    )
    word_replacement.fit(examples)

    augmentation_outputs = chain(
        perform_augmentation(
            "backtranslation",
            backtranslation,
            examples,
        ),
        perform_augmentation(
            "tfidf",
            word_replacement,
            examples,
        )
    )

    augmentation_outputs = sorted(augmentation_outputs, key=lambda x: x[0].identifier)
    augmentation_outputs = groupby(augmentation_outputs, lambda x: x[0].identifier)

    # Output to disk
    with open(output_path, "w") as file:
        for _, datapoints in augmentation_outputs:
            payload = json_dumps(
                [
                    (datapoint.text, augmentor_name)
                    for datapoint, augmentor_name in datapoints
                ]
            )
            file.write(f"{payload}\n")


def perform_augmentation(augmentor_name, augmentor, examples):
    """
    Executes the given augmentor on the specified examples

    Yields an iterator referencing the original example's index, the new augmented text, and
    the type of augmentation that was performed.

    """
    # Bundle up the datapoints that we want to augment, including their ID and how many
    # new augmentations we want to make with this technique
    augment_queue_text = [
        Datapoint(
            identifier=example_index,
            text=example_text
        )
        for example_index, example_text in enumerate(examples)
    ]

    augment_output = augmentor(augment_queue_text)

    for datapoints in augment_output:
        for datapoint in datapoints:
            yield datapoint, augmentor_name
