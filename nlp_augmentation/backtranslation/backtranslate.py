from pathlib import Path
from subprocess import call
from tempfile import NamedTemporaryFile
from math import fabs

from click import secho

from nlp_augmentation.backtranslation.postprocessor import SentToParagraph
from nlp_augmentation.backtranslation.preprocessor import SplitParagraphs
from nlp_augmentation.base import AugmentationBase
from logging import info
from uuid import uuid4
from multiprocessing import Queue, get_context
from itertools import groupby
from nlp_augmentation.data_models import SentenceDatapoint, Datapoint, AugmentedSentenceDatapoint, AugmentedDatapoint
from typing import Iterable, List
import torch
from tqdm import tqdm
from nlp_augmentation.backtranslation.worker import BackTranslateWorker


class BackTranslate(AugmentationBase):
    def __init__(
        self,
        augmentations,
        forward_model_name="transformer.wmt19.en-de.single_model",
        backward_model_name="transformer.wmt19.de-en.single_model",
        workers=1,
        use_gpu=False,
        gpu_count=1,
    ):
        """
        :param replicas: An argument for parallel preprocessing. For example, when replicas=3,
            we divide the data into three parts, and only process one part
            according to the worker_id.
        :param sampling_temp: The sampling temperature for translation. See README.md for more
            details.
        :param gpu_count: quantity of gpus to use

        """
        super().__init__(augmentations=augmentations)

        # TODO: remove model_dir
        #torch.hub.set_dir(str(model_dir))

        self.workers = workers
        self.use_gpu = use_gpu
        self.gpu_count = gpu_count

        self.forward_model_name = forward_model_name
        self.backward_model_name = backward_model_name

    def __call__(
        self,
        datapoints: Iterable[str],
        batch_size=1024,
        validate_reasonable=True
    ) -> Iterable[List[Datapoint]]:
        """
        Every time you run this component, you'll get different translated permutations
        for the same example.

        """
        datapoints = list(datapoints)

        secho("*** splitting paragraphs into sentences ***", fg="green")
        split_paragraphs = SplitParagraphs()
        sentences = list(tqdm(split_paragraphs(paragraphs=datapoints)))

        # Pre-download the models locally before we split into workers so we don't have an
        # intra-worker race condition to download the files
        secho("*** checking for forward pass weight files ***", fg="green")
        torch.hub.load("pytorch/fairseq", self.forward_model_name)

        secho("*** checking for backward pass weight files ***", fg="green")
        torch.hub.load("pytorch/fairseq", self.backward_model_name)

        secho(f"*** translating {len(sentences)} sentences ***", fg="green")

        ctx = get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()

        # Fill the queue with the inference tasks that we want to complete before launching the workers
        # to consume from this queue
        for batch in self.chunk_list(sentences, batch_size):
            input_queue.put(batch)

        workers = [
            BackTranslateWorker(
                input_queue=input_queue,
                output_queue=output_queue,
                forward_model_name=self.forward_model_name,
                backward_model_name=self.backward_model_name,
                gpu_id=worker % self.gpu_count if self.use_gpu else None,
                samples=self.augmentations,
            )
            for worker in range(self.workers)
        ]

        for worker in workers:
            worker.start()

        # (datapoint_identifier, translation_index, sentence_index, translation)
        results = []

        finished_processes = 0

        # Batches are all returned at one time, so disable smoothing so we can see the
        # average amount of datapoints computed per second interval
        with tqdm(total=len(sentences), smoothing=0) as progress_bar:
            while True:
                if finished_processes == len(workers):
                    break

                datapoint, translations = output_queue.get()

                # Exit condition for our worker
                if datapoint is None:
                    finished_processes += 1
                    continue

                results.extend(
                    AugmentedSentenceDatapoint(
                        datapoint_identifier=datapoint.datapoint_identifier,
                        sentence_index=datapoint.sentence_index,
                        augmented_index=translation_index,
                        text=translation
                    )
                    for translation_index, translation in enumerate(translations)
                )
                progress_bar.update(1)

        print("results", len(results))

        for worker in workers:
            worker.join()

        secho("*** transform sentences back into paragraphs ***", fg="green")
        sent_to_paragraph = SentToParagraph()
        paraphrases = sent_to_paragraph(results)

        if validate_reasonable:
            yield from self.select_reasonable_paraphrases(datapoints, paraphrases)
            return

        yield from paraphrases

    @staticmethod
    def chunk_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def replace_with_paraphrase(
        self, 
        ori_text,
        new_text,
        use_min_length=10,
        use_max_length_diff_ratio=0.5
    ):
        """
        Use new_text if the text length satisfies several constraints.
        """
        if len(ori_text) < use_min_length or len(new_text) < use_min_length:
            return False
    
        length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
        if fabs(length_diff_ratio) > use_max_length_diff_ratio:
            return False
    
        return True

    def select_reasonable_paraphrases(self, datapoints, translations):
        """
        Some paraphrases aren't resonable since they diverge so much from the original
        text passage in terms of length or other composition.  Limit ourselves to just
        using the ones that are valid.

        """
        for datapoint, translated_datapoints in zip(datapoints, translations):
            augmented_datapoints = []

            for translated_datapoint in translated_datapoints:
                text = (
                    translated_datapoint.text
                    if self.replace_with_paraphrase(
                        datapoint.text,
                        translated_datapoint.text,
                    )
                    else datapoint.text
                )

                augmented_datapoints.append(
                    AugmentedDatapoint(
                        identifier=datapoint.identifier,
                        augmented_index=translated_datapoint.augmented_index,
                        text=text
                    )
                )

            yield augmented_datapoints
