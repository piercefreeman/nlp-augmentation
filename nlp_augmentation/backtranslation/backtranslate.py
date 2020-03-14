from pathlib import Path
from subprocess import call
from tempfile import NamedTemporaryFile

from click import secho

from nlp_augmentation.backtranslation.postprocessor import SentToParagraph
from nlp_augmentation.backtranslation.preprocessor import SplitParagraphs


class BackTranslate:
    def __init__(self, model_dir, replicas=1, worker_id=0, sampling_temp=0.8):
        """
        :param replicas: An argument for parallel preprocessing. For example, when replicas=3,
            we divide the data into three parts, and only process one part
            according to the worker_id.
        :param sampling_temp: The sampling temperature for translation. See README.md for more
            details.

        """
        self.model_dir = str(model_dir)
        self.replicas = replicas
        self.worker_id = worker_id
        self.sampling_temp = sampling_temp

        data_dir = Path("back_trans_data")
        self.doc_len_dir = data_dir / "doc_len"
        self.forward_src_dir = data_dir / "forward_src"
        self.forward_gen_dir = data_dir / "forward_gen"
        self.backward_gen_dir = data_dir / "backward_gen"
        self.para_dir = data_dir / "paraphrase"

        data_dir.mkdir(exist_ok=True)
        self.doc_len_dir.mkdir(exist_ok=True)
        self.forward_src_dir.mkdir(exist_ok=True)
        self.forward_gen_dir.mkdir(exist_ok=True)
        self.backward_gen_dir.mkdir(exist_ok=True)
        self.para_dir.mkdir(exist_ok=True)

    def __call__(self, contents):
        with NamedTemporaryFile() as file:
            for item in contents:
                file.write(f"{item}\n".encode())
            file.seek(0)

            split_paragraphs = SplitParagraphs(
                replicas=self.replicas,
                worker_id=self.worker_id,
            )
            print(split_paragraphs)
            split_paragraphs(
                input_file=file.name,
                output_file=self.forward_src_dir / f"file_{self.worker_id}_of_{self.replicas}.txt",
                doc_len_file=self.doc_len_dir / f"doc_len_{self.worker_id}_of_{self.replicas}.json",
            )

        # TODO: Convet into native library calls
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
                f"--data_dir={self.model_dir}"
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
                f"--data_dir={self.model_dir}"
            ]
        )

        secho("*** transform sentences back into paragraphs ***", fg="green")
        sent_to_paragraph = SentToParagraph()
        return list(
            sent_to_paragraph(
                input_file=f"{self.backward_gen_dir}/file_{self.worker_id}_of_{self.replicas}.txt",
                doc_len_file=f"{self.doc_len_dir}/doc_len_{self.worker_id}_of_{self.replicas}.json",
            )
        )
