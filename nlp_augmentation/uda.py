from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
from zipfile import ZipFile

from click import group, option
from requests import get
from tqdm import tqdm

from nlp_augmentation.backtranslation.backtranslate import BackTranslate
from nlp_augmentation.word_substitution.tfidf import TfIdfWordSubstitution


## run back translation
## sent level augmentation, which calls word translation

EXAMPLES = [
    "Given the low budget and production limitations, this movie is very good.",
    "A paragraph is a group of words put together to form a group that is usually longer than a sentence. Paragraphs are often made up of several sentences. There are usually between three and eight sentences. Paragraphs can begin with an indentation (about five spaces), or by missing a line out, and then starting again. This makes it easier to see when one paragraph ends and another begins."
]

@group()
def uda():
    pass


@uda.command()
def augment():

    # randomly choose strategy
    
    if False:
        with TemporaryDirectory() as scratch_path:
            backtranslation = BackTranslate(
                model_dir=Path("~/.nlp_augmentation/checkpoints").expanduser(),
                scratch_dir=scratch_path,
            )

            paraphrased_examples = backtranslation(EXAMPLES)

            # Some paraphrases aren't resonable since they diverge so much from the original
            # text passage in terms of length or other composition.  Limit ourselves to just
            # using the ones that are valid.
            augmented_examples = backtranslation.select_reasonable_paraphrases(
                EXAMPLES,
                paraphrased_examples
            )
    else:
        word_replacement = TfIdfWordSubstitution(0.7)
        word_replacement.fit(EXAMPLES)

        print(word_replacement(EXAMPLES[0]))

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

    # Postprocessing to change a few files into the required formats
    (
        Path(checkpoints_path / "vocab.translate_enfr_wmt32k.32768.subwords")
        .rename(
            Path(checkpoints_path / "vocab.enfr.large.32768")
        )
    )
