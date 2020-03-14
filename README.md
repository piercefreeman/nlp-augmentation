# NLP Augmentation

The technique of data augmentation has shown wide success in computer vision, and early application to natural language processing as well.  This library houses plug-and-play data augmentation techniques that are intended.

## Getting Started

This code has been tested with Python 3.7.

1. Create a new virtual environment to house this project.

    ```
    python3 -m venv env nlp-transformations
    ```

1. Install the package requirements.

    ```
    pip install -e .
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

## Linting / Testing

Ensure files are linted according to flake8 and isort conventions:

```
flake8
```

To easily fix the isort errors, run:

```
isort -rc .
```