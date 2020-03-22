from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
from zipfile import ZipFile

from click import group, option, Path as ClickPath, Choice
from requests import get
from tqdm import tqdm
from json import loads as json_loads, dumps as json_dumps
from random import choices
from itertools import chain, groupby

from nlp_augmentation.backtranslation.backtranslate import BackTranslate
from nlp_augmentation.word_substitution.tfidf import TfIdfWordSubstitution


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
        "tpu",
    ]),
    required=True,
    default="cpu",
)
@option("--tpu_cloud_name", type=str, default=None)
@option("--tpu_storage_bucket", type=str, default=None)
def augment(input_path, augmentation_count, output_path, method, tpu_cloud_name, tpu_storage_bucket):
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

    translation_techniques = ["backtranslation", "tfidf"]

    # Keep track of the transformations that will be applied to each example
    # Indexed in the same order as the examples themselves
    # [[backtranslation, backtranslation, tfidf], ...]
    augmentation_per_example = [
        choices(translation_techniques, k=augmentation_count)
        for _ in range(len(examples))
    ]

    scratch_path = TemporaryDirectory()

    backtranslation_configuration_dir = {
        "gpu": {
            "use_gpu": True,
            "gpu_count": 1,
        },
        "tpu": {
            "use_tpu": True,
            "tpu_cloud_name": tpu_cloud_name,
            "tpu_storage_bucket": tpu_storage_bucket,
        }
    }

    backtranslation = BackTranslate(
        model_dir=Path("~/.nlp_augmentation/checkpoints").expanduser(),
        scratch_dir=scratch_path.name,
        **backtranslation_configuration_dir.get(method, {})
    )

    word_replacement = TfIdfWordSubstitution(0.7)
    word_replacement.fit(examples)

    augmentation_outputs = chain(
        perform_augmentation(
            "backtranslation",
            backtranslation,
            examples,
            augmentation_per_example
        ),
        perform_augmentation(
            "tfidf",
            word_replacement,
            examples,
            augmentation_per_example
        )
    )

    augmentation_outputs = sorted(augmentation_outputs, key=lambda x: x[0])
    augmentation_outputs = groupby(augmentation_outputs, lambda x: x[0])

    # Output to disk
    with open(output_path, "w") as file:
        for _, values in augmentation_outputs:
            payload = json_dumps(list(values))
            file.write(f"{payload}\n")

    # Resource cleanup
    scratch_path.cleanup()


@uda.command()
def download():
    filename = "back_trans_checkpoints.zip"
    url = f"https://storage.googleapis.com/uda_model/text/{filename}"

    request = get(url, stream=True)

    total_size = int(request.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    augmentation_path = Path("~/.nlp_augmentation").expanduser()
    checkpoints_path = augmentation_path / "checkpoints"

    # Download zip file to temporary path so if the process is inturrupted, it will
    # be automatically garbage collected
    with TemporaryFile() as file:
        for data in request.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

        progress_bar.close()
        file.seek(0)

        ZipFile(file).extractall(augmentation_path)


def perform_augmentation(augmentor_name, augmentor, examples, augmentation_per_example):
    """
    Executes the given augmentor on the specified examples

    Yields an iterator referencing the original example's index, the new augmented text, and
    the type of augmentation that was performed.

    """
    # Duplicate the relevant examples for how many unique permutations we want to create    
    # with this given augmentor
    # [Example 1, Example 1, Example 2, Example 3, Example 3, ...]
    augment_queue_index = [
        example_index
        for example_index, augmentation_types in enumerate(augmentation_per_example)
        for augmentation_type in augmentation_types
        if augmentation_type == augmentor_name
    ]

    augment_queue_text = [examples[index] for index in augment_queue_index]

    augment_output = augmentor(augment_queue_text)

    for example_index, text in zip(augment_queue_index, augment_output):
        yield example_index, text, augmentor_name
