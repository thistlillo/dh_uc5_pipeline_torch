"""Split the dataset for training.

This is the first step (00) in pipeline B.

This script splits the dataset for training, validation, and test. It is able to produce three different splits:
- random split in training, validation and test
- stratified split. Label distribution is preserved in the partitions
- k-fold stratified cross-validation.


* PLEASE NOTICE:
* UC5 pipelines are managed via a Makefile. A simple modification of a file will force make to re-run the
* entire pipeline. Remember to use "touch" to avoid unnecessary runs:
*   touch -r <oldfile> B00_prepare_training_data.py

Author: Franco Alberto Cardillo <francoalberto.cardillo@ilc.cnr.it>
Date: August 2021
"""

import fire
import numpy as np
import os
import posixpath
import pandas as pd

import common.defaults as uc5def
import common.fileutils as fu
import common.miscutils as mu
import common.partitioning as partitioning
import text.report as report

# log = mu.create_log_function("trainining-res")


class TrainingDataManager:
    def __init__(self, tsv_fld, out_fld, image_enc, train_mode, n_folders, random_seed, train_p, valid_p, log_level="info"):
        # self.conf = conf

        self.logger = uc5def.get_logger(__file__, log_level)

        self.tsv_fld = tsv_fld
        self.out_fld = out_fld
        self.image_enc = image_enc  # * we are not using the image encodings, but this assures the id will be actually found in the enc file
        self.random_seed = random_seed

        # read csv for images
        df_image_fn = posixpath.join(self.tsv_fld, image_enc)
        self.csvi = pd.read_csv(df_image_fn, sep=uc5def.csv_separator, na_filter=False) # file i_labels

        # read csv for reports
        df_reports_filename = posixpath.join(self.tsv_fld, uc5def.out_file_reports)
        self.csvr = pd.read_csv(df_reports_filename, sep=uc5def.csv_separator, na_filter=False)

        self.logger.info(f"Read csv file (images) {df_image_fn}: {len(self.csvi)} lines")
        self.logger.info(f"Read csv file (reports) {df_reports_filename}: {len(self.csvr)} lines")

        # percentages for the split
        self.training_perc = train_p
        self.valid_perc = valid_p
        self.test_perc = 1 - self.training_perc - self.valid_perc

        self.logger.info("Requested split ratios: |training|={self.training_perc:.2f}, |validation|={self.valid_perc:.2f}, |test|={self.test_perc:.2f}")
        assert 1 == self.training_perc + self.valid_perc + self.test_perc

        # ids in numpy array: used for splitting the ids into the three sets
        self.report_ids = self.csvr['id'].astype(int).to_numpy()
        self.train_ids, self.val_ids, self.test_ids = None, None, None
        self.img_training_set, self.img_test_set, self.img_validation_set = None, None, None

        # TODO: select splitting method from configuration
        self.logger.info(f'Training mode: {train_mode}')
        if train_mode == uc5def.train_mode_cv:
            self.splitter = partitioning.BalancedKFold(
                self.csvr, self.csvi, n_folders=n_folders, random_seed=self.random_seed, folder_val_perc=self.valid_perc)
        elif train_mode == uc5def.train_mode_random:
            self.splitter = partitioning.RandomSplit(self.csvr, self.csvi, self.training_perc, self.valid_perc, self.random_seed)
        else:
            self.splitter = partitioning.BalancedSplit(self.csvr, self.csvi, self.training_perc, self.valid_perc, self.random_seed)

        self.logger.info(f"Splitter: {self.splitter.get_name()}")

    # def shuffle(self):
    #     random.shuffle(self.report_ids)

    def partition_examples(self):
        n_splits = self.splitter.get_n_splits()
        exp_fld = self.out_fld
        # time_suffix = uc5def.fs_suffix()  # removed when Makefile was introduced for managing the workflow
        n_folders = self.splitter.get_n_splits()

        for i in range(n_folders):
            self.logger.info(f'preparing split {i+1}/{n_splits}')
            # folder = posixpath.join(time_suffix, uc5def.cv_folder_prefix + '_{:d}of{:d}'.format(i+1, n_folders))
            folder = posixpath.join(uc5def.cv_folder_prefix + '_{:d}of{:d}'.format(i+1, n_folders))
            self.logger.info(f'\tsubfolder for the split: {folder}')

            # # self.conf.append_suffix(folder)
            # c = copy.deepcopy(self.conf)
            # c.append_suffix(folder)

            tr, va, te = self.splitter.split(i)
            folder = posixpath.join(exp_fld, folder)

            # mu.prepare_fs(c)
            os.makedirs(folder, exist_ok=True)

            training = posixpath.join(folder, "train_i.tsv")
            test = posixpath.join(folder, "valid_i.tsv")
            validation = posixpath.join(folder, "test_i.tsv")

            # TODO: evaluate if tr, va, te should be dataframe
            # tr.to_csv(training, sep="\t")
            # va.to_csv(validation, sep="\t")
            # te.to_csv(test, sep="\t")
            np.savetxt(training, tr.astype(int), fmt='%i', delimiter=uc5def.csv_separator)
            np.savetxt(test, te.astype(int), fmt='%i',delimiter=uc5def.csv_separator)
            np.savetxt(validation, va.astype(int), fmt='%i',delimiter=uc5def.csv_separator)
            self.logger.info(f"Saved:\n\t-{training}\n\t-{validation}\n\t-{test}")
            # # c[K.image_training][K.training_set] = training
            # # c[K.image_training][K.test_set] = test
            # # c[K.image_training][K.validation_set] = validation

            # with open(posixpath.join(folder, "conf.yaml"), "w") as f:
            #     yaml.dump(c.get_conf(), f, default_flow_style=False)
        # return posixpath.join(exp_fld, time_suffix)


    # def prepare_image_set(self, report_ids):
    #     # line in image_df:
    #     #   index, report_id, image_filename, encoded MeSH
    #     # i) select reports
    #     iii = self.csvi['report'].apply(lambda x: x in report_ids)
    #     # ii) select all columns but the first two: index, report_id
    #     cols = self.csvi.columns[2:]
    #     selected = self.csvi.loc[self.csvi.index[iii], cols]
    #     return selected
    #
    #     # test
    #     # values = sorted(list(set(values)))
    #     # dd = pd.DataFrame()
    #     # dd['1'] = values
    #     # dd['2'] = sorted(ids)
    #     # dd.to_csv('/home/cardillo/data/deephealth/train.csv')
    #     # print(dd)


    # # TODO: this method is not used in current implementation
    # def save(self):
    #     settings = self.conf  # self.conf[K.experiment_settings]
    #     name = settings[K.exp_name]

    #     exp_folder = self.conf.out_folder(K.experiment_fld)
    #     os.makedirs(exp_folder, exist_ok=True)

    #     # if the configuration specified values for training/validation/test sets
    #     #    it means that the user wants to use a specific split and that this class
    #     #    has been run as part of a standard pipeline.
    #     # In this case we do not save the split and do not modify the values for
    #     # the training data.
    #     # Otherwise, we save the split and store the paths in the appropriate fields
    #     # in configuration.
    #     img_settings = settings[K.image_training]
    #     paths_specified = img_settings[K.training_set] or img_settings[K.validation_set] or img_settings[K.test_set]

    #     if not paths_specified:
    #         training = posixpath.join(exp_folder, "train_i.tsv")
    #         test = posixpath.join(exp_folder, "valid_i.tsv")
    #         validation = posixpath.join(exp_folder, "test_i.tsv")
    #         self.img_training_set.to_csv(training, sep="\t")
    #         self.img_validation_set.to_csv(test, sep="\t")
    #         self.img_test_set.to_csv(validation, sep="\t")
    #         img_settings[K.training_set] = training
    #         img_settings[K.test_set] = test
    #         img_settings[K.validation_set] = validation
    #     else:
    #         log("The paths to training-validation-test specified in conf - NOT overwriting them")

    #     conf_fn = posixpath.join(exp_folder, "conf.yaml")
    #     with open(conf_fn, "w") as f:
    #         yaml_conf = self.conf.get_conf()
    #         print('Saving YAML configuration:', conf_fn)
    #         yaml.dump(yaml_conf, f, default_flow_style=False)

    #     log("Training data saved in folder: %s" % exp_folder)
    #     return exp_folder

def main(tsv_fld, out_fld, image_enc, train_mode, n_folders, random_seed, train_p, valid_p, log_level="info"):
    # TODO: skip this step if conf contains paths to pre-existing splits
    # with open(args.conf, 'r') as stream:
    #     conf = yaml.safe_load(stream)
    mng = TrainingDataManager(tsv_fld, out_fld, image_enc, train_mode, n_folders, random_seed, train_p, valid_p)
    fld = mng.partition_examples()
    return fld
    # fld = mng.save()
    # log(f'Experiment folder: {fld}')

    
if __name__ == "__main__":
    fire.Fire(main)
