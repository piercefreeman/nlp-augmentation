from pathlib import Path
from subprocess import call
from tempfile import NamedTemporaryFile
from math import fabs

from click import secho

from nlp_augmentation.backtranslation.postprocessor import SentToParagraph
from nlp_augmentation.backtranslation.preprocessor import SplitParagraphs
from nlp_augmentation.base import AugmentationBase
from logging import info


class BackTranslate(AugmentationBase):
    def __init__(self, model_dir, scratch_dir, replicas=1, worker_id=0, sampling_temp=0.8, gpu_count=0):
        """
        :param replicas: An argument for parallel preprocessing. For example, when replicas=3,
            we divide the data into three parts, and only process one part
            according to the worker_id.
        :param sampling_temp: The sampling temperature for translation. See README.md for more
            details.
        :param gpu: quantity of gpus to use

        """
        self.model_dir = str(model_dir)
        self.replicas = replicas
        self.worker_id = worker_id
        self.sampling_temp = sampling_temp
        self.gpu_count = gpu_count

        scratch_dir = Path(scratch_dir)
        self.doc_len_dir = scratch_dir / "doc_len"
        self.forward_src_dir = scratch_dir / "forward_src"
        self.forward_gen_dir = scratch_dir / "forward_gen"
        self.backward_gen_dir = scratch_dir / "backward_gen"
        self.para_dir = scratch_dir / "paraphrase"

        self.doc_len_dir.mkdir(exist_ok=True)
        self.forward_src_dir.mkdir(exist_ok=True)
        self.forward_gen_dir.mkdir(exist_ok=True)
        self.backward_gen_dir.mkdir(exist_ok=True)
        self.para_dir.mkdir(exist_ok=True)

    def __call__(self, examples, validate_reasonable=True):
        """
        Every time you run this component, you'll get different translated permutations
        for the same example.

        # TODO: Add function determinism for test cases.

        """
        with NamedTemporaryFile() as file:
            for item in examples:
                file.write(f"{item}\n".encode())
            file.seek(0)

            split_paragraphs = SplitParagraphs(
                replicas=self.replicas,
                worker_id=self.worker_id,
            )
            split_paragraphs(
                input_file=file.name,
                output_file=self.forward_src_dir / f"file_{self.worker_id}_of_{self.replicas}.txt",
                doc_len_file=self.doc_len_dir / f"doc_len_{self.worker_id}_of_{self.replicas}.json",
            )

        # TODO: Convert into native library calls
        # https://github.com/tensorflow/tensor2tensor/blob/c1165f67966b86d9fa304ef8d1b745f70a7b9f75/tensor2tensor/bin/t2t_decoder.py
        secho("*** forward translation ***", fg="green")
        call(
            [
                "t2t-decoder",
                "--problem=translate_enfr_wmt32k",
                "--model=transformer",
                "--hparams_set=transformer_big",
                f"--hparams=sampling_method=random,sampling_temp={self.sampling_temp}",
                "--decode_hparams=beam_size=1,batch_size=16",
                f"--checkpoint_path={self.model_dir}/enfr/model.ckpt-500000",
                "--output_dir=/tmp/t2t",
                f"--decode_from_file={self.forward_src_dir}/file_{self.worker_id}_of_{self.replicas}.txt",
                f"--decode_to_file={self.forward_gen_dir}/file_{self.worker_id}_of_{self.replicas}.txt",
                f"--data_dir={self.model_dir}",
                f"--worker_gpu={self.gpu_count}",
            ]
        )

        secho("*** backward translation ***", fg="green")
        call(
            [
                "t2t-decoder",
                "--problem=translate_enfr_wmt32k_rev",
                "--model=transformer",
                "--hparams_set=transformer_big",
                f"--hparams=sampling_method=random,sampling_temp={self.sampling_temp}",
                "--decode_hparams=beam_size=1,batch_size=16,alpha=0",
                f"--checkpoint_path={self.model_dir}/fren/model.ckpt-500000",
                "--output_dir=/tmp/t2t",
                f"--decode_from_file={self.forward_gen_dir}/file_{self.worker_id}_of_{self.replicas}.txt",
                f"--decode_to_file={self.backward_gen_dir}/file_{self.worker_id}_of_{self.replicas}.txt",
                f"--data_dir={self.model_dir}",
                f"--worker_gpu={self.gpu_count}",
            ]
        )

        secho("*** transform sentences back into paragraphs ***", fg="green")
        sent_to_paragraph = SentToParagraph()
        paraphrases = list(
            sent_to_paragraph(
                input_file=f"{self.backward_gen_dir}/file_{self.worker_id}_of_{self.replicas}.txt",
                doc_len_file=f"{self.doc_len_dir}/doc_len_{self.worker_id}_of_{self.replicas}.json",
            )
        )

        if validate_reasonable:
            return self.select_reasonable_paraphrases(examples, paraphrases)
        return paraphrases

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

    def select_reasonable_paraphrases(self, examples, paraphrases):
        """
        Some paraphrases aren't resonable since they diverge so much from the original
        text passage in terms of length or other composition.  Limit ourselves to just
        using the ones that are valid.

        """
        assert len(examples) == len(paraphrases)

        return [
            (
                paraphrase
                if self.replace_with_paraphrase(
                    example,
                    paraphrase,
                )
                else example
            )
            for example, paraphrase in zip(examples, paraphrases)
        ]
