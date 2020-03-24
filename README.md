# NLP Augmentation

The technique of data augmentation has shown wide success in computer vision and early application to natural language processing as well.  This library houses various modules for the latest NLP data augmentation techniques.  It's intended to be as easy as possible to pass your own dataset and get back high-quality datapoints to augment your original data for use in semi-supervised learning.

## Getting Started

This code has been tested with Python 3.7.

1. Create a new virtual environment to house this project.

    ```
    python3 -m venv env nlp-transformations
    ```

1. Install the package requirements.  Run:

    ```
    pip install -e .
    ```

1. Install `nltk` dependencies:

    ```
    python -m nltk.downloader punkt
    ```

## UDA

Originally from [Google Research](https://github.com/google-research/uda).

1. Either import the uda augmentation classes manually into a separate python project, or leverage the CLI to supplement a `.jsonl` file that's on disk.  Format each line with a datapoint that includes a `text` key with the text that you'd like to translate.

    ```
    {"text": "This is an example sentence to translate."}
    {"text": "This is another sentence to translate."}
    ```

    ```
    uda augment --input-path ./example.jsonl --output-path ./example-supplemented.jsonl --augmentation-count 5 [--method gpu --gpu-count 1] [--workers 2]
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