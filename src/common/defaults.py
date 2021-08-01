import logging
import random
import time
import torch

folder_images = 'images'
folder_texts = 'texts'
folder_output = 'output'
folder_encodings = 'encoded'
folder_experiments = 'experiment'

csv_extension = ".tsv"  # since we are dealing with text, prefer \t in place of ,
tmp_extension = ".tmp"  # appended to filenames for files written during long computations that might fail

csv_separator = "\t"
seq_separator = " ; "  # used for separating multiple values in the same csv column

# for reproducibility
random_seed = 100
random_seed_alt = 232

# files produced in the first reading of the dataset
# no checks on existing images, text saved as in the reports, etc.
out_file_reports_raw = "reports_raw" + csv_extension
out_file_images_raw = "images_raw" + csv_extension

# first pre-processing:
# - missing images removed
# - reports without images removed
# NOTICE in the std-dataset all reports have images associated and all the referred images exist
out_file_reports_ver = "reports_ver" + csv_extension
out_file_images_ver = "images_ver" + csv_extension

# lower case text, produced starting from _ver
out_file_reports_lower = "reports_lower" + csv_extension
out_file_images_lower = "images_lower" + csv_extension

# these names are to be used for cleaned-versions or, in any case,
#    for the versions used in the splitting into: findings.tsv, indications.tsv, etc.
# These two files are obtained from the _lower files, i.e., all referred images exist, etc.
out_file_reports = "reports" + csv_extension
out_file_images = "images" + csv_extension


# csv_mesh_img = "mesh_img"
csv_img_mesh = "img_mesh_dense" 
csv_mesh_freq = "mesh_freq"
csv_termx_given_termy = "termx_given_termy"

enc_folder = "encoded"
enc_term_list = "mesh_term_list"
enc_mesh_terms = "e_mesh_terms"
enc_image_terms = "e_image_labels"
enc_image_terms_adjusted = "e_image_labels_norm_sm"

# misc
cv_folder_prefix = "fld"
timestamp_fmt_fs = "%Y%m%d-%H%M%S"
train_mode_random = 'random'
train_mode_cv = 'cross_validation'
train_mode_standard = 'standard'

def fs_suffix():
    return time.strftime(timestamp_fmt_fs)


log_levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}
def get_logger(name=None,level="info"):
    if level is None:
        level = "info"
        
    logging.basicConfig(
        #     filename='src.log',
        level=log_levels[level.lower()],
        format='%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(name)


def manual_seed(seed: int) -> None:
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed: Random state seed

    .. versionchanged:: 0.4.3
        Added ``torch.cuda.manual_seed_all(seed)``.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
