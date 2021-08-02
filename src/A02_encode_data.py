"""Encode image labels and mesh terms (also referred to as tags in subsequent steps) as 1-hot vectors.

This is the third step (02, after 00, 01) in pipeline A.


* PLEASE NOTICE:
* UC5 pipelines are managed via a Makefile. A simple modification of a file will force make to re-run the
* entire pipeline. Remember to use "touch" to avoid unnecessary runs:
*   touch -r <oldfile> A02_encode_data.py

Author: Franco Alberto Cardillo <francoalberto.cardillo@ilc.cnr.it>
Date: July 2021
"""

import fire
import numpy as np
import os
import pandas as pd
import posixpath
from tqdm import tqdm

import common.defaults as uc5def
import common.miscutils as mu

from encoders.one_hot_encoder import OneHotEncoder
from encoders.image_encoder import ImageEncoder

logger = None

def get_unique_mesh_terms(df, col):
    global logger
    # major_mesh in dataframe contains multiple terms separated
    # by a semicolon
    terms = df[col].unique().tolist()
    # expand sequences of major mesh terms, split on semicolon
    terms = [x for t in terms for x in t.split(uc5def.seq_separator)]

    # NOTICE:
    uset = set(terms)  # this set contains 1679 terms
    logger.debug("Unique terms in %s: %d" % (col, len(uset)))

    # many strings in terms are composed by multiple words separated by '/'
    # we split such strings and keep the single tokens
    uset2 = set([x for t in terms for x in t.split("/")])
    unique_list = sorted(list(uset2))
    logger.debug("Unique terms, 2nd split: %d" % len(unique_list))
    # for elem in uset_list:
    #     log(elem)

    # NOTICE: some terms are composed by multiple tokens, separated by comma
    # TODO: evaluate third split on comma ,
    return unique_list


def build_img_to_mesh_bin_mat(images_df, mesh_l):
    # img_terms_df is n_images rows x n_mesh_terms cols
    img_terms_df = pd.DataFrame(np.zeros(shape=(len(images_df), len(mesh_l)), dtype=bool), columns=mesh_l)
    term_frequency = pd.DataFrame()  # shape=(len(mesh_l), 2), dtype=(object, int), columns=['term', 'count'])

    with tqdm(total=len(images_df)) as pbar:
        for row_i in range(len(images_df)):
            row = images_df.iloc[row_i]
            # TODO:
            # list of terms that should? be ignored: "no indexing"
            # single_terms = [t for x in major_mesh.split(uc5def.seq_separator) for t in x.split('/')]
            single_terms = row['major_mesh_l']

            img_terms_df.loc[row_i, single_terms] = True
            pbar.update(1)

    img_terms_df.insert(0, 'image_filename', images_df.image_filename)

    term_frequency['freq'] = img_terms_df[mesh_l].sum(axis=0)
    term_frequency['term'] = mesh_l
    return {uc5def.csv_img_mesh: img_terms_df, uc5def.csv_mesh_freq: term_frequency}


def build_pairw_cond_prob_mat(img_mesh_df, uterms_l):
    print(img_mesh_df.columns)
    counts = np.zeros(shape=(len(uterms_l), len(uterms_l)), dtype=float)
    with tqdm(total=len(uterms_l) * (len(uterms_l) - 1)) as pbar:
        for i, term in enumerate(uterms_l):
            iii = img_mesh_df[term]  # == True)
            n_times = np.count_nonzero(iii)

            for j, term2 in enumerate(uterms_l):
                if i == j:
                    continue
                jjj = img_mesh_df[term2]
                kkk = (iii & jjj)
                n_coocc = np.count_nonzero(kkk)
                counts[i, j] = n_coocc / n_times
                # if counts[i, j] == 1:
                #     print("%s, %s: %d / %d = %.2f" % (term, term2, n_times, n_coocc, float(n_coocc) / n_times))
            pbar.update(len(uterms_l) - 1)
    df = pd.DataFrame(data=counts, columns=uterms_l)
    df.insert(0, 'term', uterms_l)
    df.set_index('term', inplace=True)
    print(df.head)
    return df


def save_results(dfs, folder):
    global logger
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    for k, df in dfs.items():
        filename = posixpath.join(folder, k) + uc5def.csv_extension
        df.to_csv(filename, sep=uc5def.csv_separator)
        logger.info('Saved: %s' % filename)


def main(in_tsv_fld, out_fld, log_level="info"):
    """Script for encoding the image labels. 
    The encoding could be performed directly in the torch dataloader. However, since this use case will
    be implemented also using the EDDL library, the encoding has been implemented in the pre-processing pipeline
    and shared between the PyTorch and the EDDL implementations.
    
    Args:
        in_tsv_fld (string): input folder containing the tsv files.
        out_fld (string): output folder for the tsv files.
        log_level (string): log level as defined in python logging module (without the "logging." prefix). Default, info.
    """
    
    global logger
    logger = uc5def.get_logger(__file__, log_level)
    
    fld = in_tsv_fld
    filename = posixpath.join(fld, uc5def.out_file_reports)

    df = pd.read_csv(filename, sep=uc5def.csv_separator, na_filter=False)
    mu.add_list_of_mesh(df)

    logger.info("file %s read. shape %s" % (filename, df.shape))
    logger.info("\tcolumns: %s" % str(df.columns))

    # list of unique mesh terms
    uterms_l = get_unique_mesh_terms(df, 'major_mesh')
    
    term_encoder = OneHotEncoder(uterms_l, add_out_of_dict=False, ignore_key_errors=True)
    image_encoder = ImageEncoder(df, term_encoder)

    term_encoding = term_encoder.get_encoding_df()
    image_encoding = image_encoder.get_encoding_df()
    
    save_results({uc5def.enc_mesh_terms: term_encoding, uc5def.enc_image_terms: image_encoding}, out_fld)
    logger.info("encoding completed with success")


if __name__ == "__main__":
    fire.Fire(main)