import torch
from multiprocessing.context import SpawnProcess
from itertools import groupby
from queue import Empty
from random import choices, sample
from click import secho


class BackTranslateWorker(SpawnProcess):
    def __init__(
        self,
        input_queue,
        output_queue,
        forward_model_name,
        backward_model_name,
        gpu_id=None,
        samples=1,
    ):
        super().__init__()

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.forward_model_name = forward_model_name
        self.backward_model_name = backward_model_name
        self.gpu_id = gpu_id
        self.samples = samples

        self.forward_model = None
        self.backward_model = None

    def init_models(self):
        self.forward_model = torch.hub.load("pytorch/fairseq", self.forward_model_name)
        self.backward_model = torch.hub.load("pytorch/fairseq", self.backward_model_name)

        # TODO: Consider memory impact of adding multiple models to the GPU at the same time
        if self.gpu_id is not None:
            self.forward_model.cuda(torch.device(f"cuda:{self.gpu_id}"))
            self.backward_model.cuda(torch.device(f"cuda:{self.gpu_id}"))

    def run(self):
        batch = None

        try:
            # When process is spawned, init the translation models
            # Note that we don't currently share memory space, which could optimize performance
            # on CPU architectures
            # https://pytorch.org/docs/stable/multiprocessing.html
            self.init_models()

            while True:
                try:
                    batch = self.input_queue.get(block=False)
                except Empty:
                    self.output_queue.put((None, None))
                    return

                forward_results = self.run_forward_translation(batch)
                backward_results = self.run_backward_translation(forward_results)

                for datapoint, translations in zip(batch, backward_results):
                    self.output_queue.put((datapoint, translations))
        except RuntimeError as e:
            secho(e, fg="red")

            # Worker failed (likely failed to initialize memory), add this batch back
            # into the queue since it was incomplete
            if batch is not None:
                self.input_queue.put(batch)

            # Worker will end
            self.output_queue.put((None, None))

    def run_forward_translation(self, batch):
        # "Hello world!" -> 'Hello world !'
        binarized_tokens = [
            self.forward_model.encode(datapoint.text)
            for datapoint in batch
        ]
 
        # We do the augmentation upscaling (1 datapoint -> k backtranslations) via
        # the forward language pass
        # Generate the requested numbers of translations via top-k sampling
        # [examples, quantity samples, ...]
        return self.forward_model.generate(binarized_tokens, **self.sample_hyperparameters)

    def run_backward_translation(self, forward_results):
        # Generate the examples to translate during the backward pass
        backward_binarized_tokens = [
            (datapoint_id, datapoint_sample["tokens"])
            for datapoint_id, datapoint in enumerate(forward_results)
            for datapoint_sample in self.autosize_list(datapoint, self.samples)
        ]

        backward_text_input = [tokens.cpu() for _, tokens in backward_binarized_tokens]
        backward_indices_input = [identifier for identifier, _ in backward_binarized_tokens]

        backward_results = self.backward_model.generate(backward_text_input, **self.sample_hyperparameters)

        # Re-group the backtranslated results by their original batch id
        translations_group = [
            [
                self.backward_model.decode(backward_result["tokens"])
                for _, backward_results in group
                for backward_result in backward_results
            ]
            for _, group
            in groupby(
                zip(backward_indices_input, backward_results),
                key=lambda x: x[0]
            )
        ]

        # Crop to our sample size
        sampled_translations = [
            self.autosize_list(translations, self.samples)
            for translations in translations_group
        ]

        return sampled_translations

    @property
    def sample_hyperparameters(self):
        return dict(
            beam=int(self.samples / 2 + 0.5),
            sampling=True,
            sampling_topk=50,
            temperature=1.2,
        )

    def autosize_list(self, original_list, size):
        """
        Crop a list to a given size.  If the desired size is greater than the size of list,
        will randomly fill the remaining space from elements in original_list.

        """
        new_list = sample(original_list, min(len(original_list), size))
        new_list += choices(original_list, k=(size - len(new_list)))
        return new_list
