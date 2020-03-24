import torch
from multiprocessing.context import SpawnProcess
from itertools import groupby
from queue import Empty


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
        return self.forward_model.generate(
            binarized_tokens,
            beam=self.samples,
            sampling=True,
            sampling_topk=self.samples*4
        )

    def run_backward_translation(self, forward_results):
        # Generate the examples to translate during the backward pass
        backward_binarized_tokens = [
            (datapoint_id, sample["tokens"])
            for datapoint_id, datapoint in enumerate(forward_results)
            for sample in datapoint
        ]

        backward_text_input = [tokens.cpu() for _, tokens in backward_binarized_tokens]
        backward_indices_input = [identifier for identifier, _ in backward_binarized_tokens]

        # Backward results with static beam search size to allow for some consideration
        # of high-probability results
        # Here we do a 1:1 backtranslation into the original language
        backward_results = self.backward_model.generate(backward_text_input, beam=1)

        # Re-group the backtranslated results by their original batch id
        return [
            [
                self.backward_model.decode(backward_result[0]["tokens"])
                for _, backward_result in group
            ]
            for _, group
            in groupby(
                zip(backward_indices_input, backward_results),
                key=lambda x: x[0]
            )
        ]

