# NLP Augmentation

The technique of data augmentation has shown wide success in computer vision, and early application to natural language processing as well.  This library houses plug-and-play data augmentation techniques that are intended.

## Getting Started

This code has been tested with Python 3.7.

1. Create a new virtual environment to house this project.

    ```
    python3 -m venv env nlp-transformations
    ```

1. Install the package requirements.  If you are using a CPU, run:

    ```
    pip install -e .[cpu]
    ```

    If you'd like it optimized for your GPU, use:

    ```
    pip install -e .[gpu]
    ```

1. Install `nltk` dependencies:

    ```
    python -m nltk.downloader punkt
    ```

Had to rename subwords vocab file in `/checkpoints` to `vocab.enfr.large.32768`

## UDA

Originally from [Google Research](https://github.com/google-research/uda).

1. Install the pre-trained translation models that allow UDA to forward-translate text passages and back-translate them into the original language:

    ```
    uda download
    ```

    This download places the models within `~/.nlp_augmentation` to cache them across multiple virtualenvs.

1. Either import the uda augmentation classes manually into a separate python project, or leverage the CLI to supplement a `.jsonl` file that's on disk.  Format each line with a datapoint that includes a `text` key with the text that you'd like to translate.

    ```
    {"text": "This is an example sentence to translate."}
    {"text": "This is another sentence to translate."}
    ```

    ```
    uda augment --input-path ./example.jsonl --output-path ./example-supplemented.jsonl --augmentation-count 5 [--gpu-count 1] [--use-tpu True --cloud_tpu_name XXX]
     ```

## Linting / Testing

Ensure files are linted according to flake8 and isort conventions:

```
flake8
```

To easily fix the isort errors, run:

```
isort -rc .
```