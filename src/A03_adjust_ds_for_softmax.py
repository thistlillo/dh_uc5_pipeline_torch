"""Normalize the 1-hot encoding vectors of the images by their L1 norm.

This is the fourth step (03, after 00, ..., 02) in pipeline A.


* PLEASE NOTICE:
* UC5 pipelines are managed via a Makefile. A simple modification of a file will force make to re-run the
* entire pipeline. Remember to use "touch" to avoid unnecessary runs.

Author: Franco Alberto Cardillo <francoalberto.cardillo@ilc.cnr.it>
Date: July 2021
"""
import fire
import numpy as np
import pandas as pd
from posixpath import join

import common.defaults as uc5def

def main(in_tsv_fld, out_fld, log_level="info", dev_mode=False):
    """Script for the alternative encoding the image labels. 
    Basically, one hot vectors are normalized by their L1 norm. With this encoding, subsequent NN module can use softmax + cross entropy loss.
    
    The encoding could be performed directly in the torch dataloader. However, since this use case will
    be implemented also using the EDDL library, the encoding has been implemented in the pre-processing pipeline
    and shared between the PyTorch and the EDDL implementations to simplify their dataloaders.
    
    Args:
        in_tsv_fld (string): input folder containing the tsv files.
        out_fld (string): output folder for the tsv files.
        log_level (string): log level as defined in python logging module (without the "logging." prefix). Default, info.
        dev_mode (boolean): perform a check on the result, useful when chaning the input encodings. Default, False.
    """
    in_images = join(in_tsv_fld, uc5def.enc_image_terms) + uc5def.csv_extension
    out_images = join(out_fld, uc5def.enc_image_terms_adjusted) + uc5def.csv_extension

    logger = uc5def.get_logger(__file__, log_level)
    logger.info(f'input: {in_images}, output: {out_images}')
    if dev_mode:
        logger.warning('development mode ON')

    df = pd.read_csv(in_images, sep='\t')

    sc = 3 # start (first) column
    values = df.iloc[:, sc:].values

    nl = 1.0 / np.count_nonzero(values, axis=1)
    nl = nl.reshape(-1, 1)

    df.iloc[:, sc:] = np.multiply(values, nl)
    df.to_csv(out_images, sep='\t', index=False)
    if dev_mode:
        logger.warning('the following message should NOT indicate an empty list')
        values = df.iloc[:, sc:].values
        l = values[(values > 0) & (values < 1)]
        logger.warning(f'number of values: {len(l)} <- THIS SHOULD BE > 0')
        assert(len(l)>0)

    logger.info(f'saved {out_images}')
    logger.info("alternative encoding built with success")


if __name__ == "__main__":
    fire.Fire(main)
