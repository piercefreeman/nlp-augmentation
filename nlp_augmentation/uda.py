from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile

from click import group, option
from requests import get
from tqdm import tqdm

from nlp_augmentation.backtranslation.backtranslate import BackTranslate


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
    # Every time you run this component, you'll get different permutations for the same example
    backtranslation = BackTranslate(
        model_dir=Path("~/.nlp_augmentation/checkpoints").expanduser()
    )
    paraphrased_examples = backtranslation(EXAMPLES)

    aa


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

""" 
def proc_and_save_unsup_data(
    processor, sub_set,
    raw_data_dir, data_stats_dir, unsup_out_dir,
    tokenizer,
    max_seq_length, trunc_keep_right,
    aug_ops, aug_copy_num,
    worker_id, replicas
):
    # print random seed just to double check that we use different random seeds
    # for different runs so that we generate different augmented examples for the same original example.
    random_seed = np.random.randint(0, 100000)
    tf.logging.info("random seed: {:d}".format(random_seed))
    np.random.seed(random_seed)
    tf.logging.info("getting examples")

    elif sub_set.startswith("unsup"):
        ori_examples = processor.get_unsup_examples(raw_data_dir, sub_set)

    # this is the size before spliting data for each worker
    data_total_size = len(ori_examples)

    # We assume right now that we're running on one worker
    # TODO: Parallelize with multiprocessing queues as appropriate
    start = 0
    end = len(ori_examples)

  tf.logging.info("getting augmented examples")
  aug_examples = copy.deepcopy(ori_examples)
  aug_examples = sent_level_augment.run_augment(
      aug_examples, aug_ops, sub_set,
      aug_copy_num,
      start, end, data_total_size)

    # Yield these as examples for each response


# Preprocess unlabeled set
python preprocess.py \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=data/proc_data/IMDB/unsup \
  --back_translation_dir=data/back_translation/imdb_back_trans \
  --data_type=unsup \
  --sub_set=unsup_in \
  --aug_ops=bt-0.9 \
  --aug_copy_num=0 \
  --vocab_file=$bert_vocab_file \
  $@




    tf.logging.info("Create unsup. data: subset {} => {}".format(
        FLAGS.sub_set, unsup_out_dir))
    proc_and_save_unsup_data(
        processor, FLAGS.sub_set,
        FLAGS.raw_data_dir, data_stats_dir, unsup_out_dir,
        tokenizer, FLAGS.max_seq_length, FLAGS.trunc_keep_right,
        FLAGS.aug_ops, FLAGS.aug_copy_num,
        FLAGS.worker_id, FLAGS.replicas)
 """
