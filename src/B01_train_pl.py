
import fire
import glob
import logging
import os
from pytorch_lightning.loggers import NeptuneLogger
from posixpath import join
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from common.defaults import init_logging
from common.FakeRun import FakeRun
from torch_impl.data.datamod import UC5DataModule
import common.dataloading as dataloading
from torch_impl.models.uc5_model import Uc5Model

def get_cnn():
    return None

def get_sent_lstm():
    return None

def get_word_lstm():
    return None


def main(tsv_fld,
         image_fld,
         exp_fld,
         conf,
         image_encoding,
         cnn="vgg19",
         batch_size=20,
         learning_rate=1e-5,
         momentum=0.9,
         n_epochs=100,
         max_sentences=3,
         max_words=10,
         early_stopping=None,
         stop_iteration=None,
         enable_amp=False,
         enable_gpu=True,
         enable_tensorboard=False,
         enable_remotelog=False,
         seed=1,
         log_level=logging.INFO,
         log_every_iters=20,
         backend=None,
         nproc_per_node=None,
         num_workers=1,
         dev_mode=False,
         tag_emb_size = 512, # dim embeddings semantic feaures, should be 512
         tag_topk = 5, # top k tags to forward
         attn_emb_size = 512, # embedding dimension of the co-attention network
         lstm_sent_h_size = 512,  # size hidden state of sentence LSTM
         lstm_sent_dropout = 0,
         lstm_sent_n_layers = 2,
         lstm_word_h_size = 512,
         lstm_word_dropout = 0,
         lstm_word_n_layers = 1,
         loss_lambda_tag = 10000,
         loss_lambda_stop = 10,
         loss_lambda_word = 1,
         **other_kwargs
         ):
    """Script for training a VGG model on the Chest X-Ray dataset (UC5, DeepHealth)
    
    Args:
        tsv_fld (string): path to the folder containing tsv files
        img_fld (string): path to the folder containining the images
        exp_fld (string): folder with data for the experiments, such as, e.g., the split of the examples
        conf (string): path to the YAML configuration file
        image_encoding (string): path to the tsv file with the (encoded) training labels
        cnn (string): vgg model to train, see file models/vgg.py for the available models. Default: "VGG19".
        batch_size (int): batch size. Default, 10.
        learning_rate (float): learning rate Default, 1e-3.
        momentum (float): momentum. Default, 0.9.
        n_epochs (int): Maximum number of training epochs. Default, 100.
        max_sentences (int): Maximum number of sentences to learn (and to generate). Default, 5.
        max_words (int): Maximum number of words to keep in a sentence. Default, 10.
        early_stopping (string): Criterion for early stopping. Default, None.
        stop_iteration (int, optional): iteration to stop the training. Can be used to check resume from checkpoint. Default, None.
        enable_amp (boolean): enable automatic mixed precision. Default, False
        enable_gpu (boolean): enable computations on the GPU. Default, True.
        enable_tensorboard (boolean): enable TensorBoard (default: False)
        enable_remotelog (boolean): enable remote logging on Neptune.ai. Default, False.
        seed (int): seed for the generator of random numbers. Default, 1.
        log_level (boolean): level of the logger in the python logging module. Default, logging.INFO.
        log_every_iters (int): log batch loss every ``log_every_iters`` iterations. Default, 20.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes. Default, None.
        num_workers (int): number of workers in the data loader. Default, 1.
        dev_mode (boolean): activate dev_mode, limit number of train/valid/test epochs. Default, False.
        **other_kwargs: other
    """
    config = locals()

    init_logging()
    logger = logging.getLogger("C01_train_pl")
    eval(f"logger.setLevel(logging.{config['log_level'].upper()})")

    neptune_logger, neptune_logger_pl = None, None
    
    if enable_remotelog:
        logger.info("Remote logging enabled")
        neptune_logger = NeptuneLogger(
            project='thistlillo/UC5-DeepHealth-PyTorch',
            name='lightning-run',  # Optional
        neptune_logger_pl = neptune_logger  # this one will be passed to pytorch lighting, the previous will be used
        )
    else:
        logger.info("Remote logging disabled, using a fake logger")
        neptune_logger = FakeRun()
        
    # process exp folder and see that subfolders are there
    subfolders = glob.glob(f'{exp_fld}/fld*')
    logger.debug(f'About to launch |number of training|={len(subfolders)}')

    for subf_i, subf in enumerate(subfolders):
        path = join(exp_fld, subf)
        checkpoints_dir = join(path, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        logger.info(f"Accessing subfolder {subf_i+1}/{len(subfolders)}: {path}")
        config['output_fld'] = path
        # launch pytorch-lightning trainer
        model = Uc5Model(config, local_logger=logger, remote_logger=neptune_logger_pl)
        # print(model)
        checkpoint_callb = ModelCheckpoint(monitor="avg_val_loss",
                                           dirpath=checkpoints_dir,
                                           filename=config['cnn'] +
                                           "_{epoch:02d}_{valloss:.2f}",
                                           mode="min")

        limit_train_batches = 1
        if config["dev_mode"]:
            limit_train_batches = 0.05

        limit_val_batches = limit_train_batches
        limit_test_batches = limit_train_batches

        trainer = pl.Trainer(gpus=[0,3],
                             # accelerator="ddp",
                             precision=16,
                             max_epochs=config["n_epochs"],
                             progress_bar_refresh_rate=10,
                             callbacks=[checkpoint_callb],
                             limit_train_batches=limit_train_batches,
                             limit_val_batches=limit_val_batches,
                             limit_test_batches=limit_test_batches,
                             logger=neptune_logger_pl
                             )

        # img_enc_df, df, img_dir, data_dir, img_transform=None, logger=None
        data = UC5DataModule(config, logger=logger)
        trainer.fit(model, datamodule=data)
        logger.info("Fit completed")
    logger.info("C01_train_pl complete. Exiting with success..")

if __name__ == "__main__":
    fire.Fire(main)
