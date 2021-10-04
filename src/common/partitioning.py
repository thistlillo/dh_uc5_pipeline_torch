from datetime import datetime
import math
import numpy as np
import pandas as pd
import random

import common.miscutils as mu
import common.defaults as uc5def

log = mu.create_log_function("partitioning")

class BalancedSplit:
    def __init__(self, reports, images, train_perc, valid_perc, random_seed=datetime.now()):
        self.csvr = reports
        self.csvi = images
        self.random_seed = random_seed
        random.seed(random_seed)
        self.n_reports = len(reports)
        self.n_images = len(images)
        self.report_ids = self.csvr['id'].to_numpy(dtype=int)
        self.train_ids, self.val_ids, self.test_ids = None, None, None
        self.img_training_set, self.img_test_set, self.img_validation_set = None, None, None
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.test_perc = 1 - self.train_perc - self.valid_perc

    def get_name(self):
        return 'balanced_split'

    def get_n_splits(self):
        return 1

    def split(self, split_idx=0):
        test_perc = 1 - self.train_perc - self.valid_perc
        r = self.csvr
        # partition: positive and negative examples
        iii = r.major_mesh == 'normal'
        neg_ids = r.loc[iii, 'id'].to_numpy(dtype=int)  # r.index[iii].to_numpy(dtype=int)
        pos_ids = r.loc[~iii, 'id'].to_numpy(dtype=int)  # r.index[~iii].to_numpy(dtype=int)
        print(f'|reports|={len(r)}, |neg_ids|={len(neg_ids)}, |pos_ids|={len(pos_ids)}')
        random.shuffle(neg_ids)
        random.shuffle(pos_ids)
        neg_train, neg_val, neg_test = self.inner_split(neg_ids)
        pos_train, pos_val, pos_test = self.inner_split(pos_ids)
        train_ids = np.concatenate((neg_train, pos_train), axis=None).astype(int)  # , dtype=int)
        val_ids = np.concatenate((neg_val, pos_val), axis=None).astype(int)  # , dtype=int)
        test_ids = np.concatenate((neg_test, pos_test), axis=None).astype(int)  # ), dtype=int)

        random.shuffle(train_ids)
        random.shuffle(val_ids)

        # random.shuffle(test_ids)
        # return indexes of reports
        return train_ids, val_ids, test_ids


    def inner_split(self, report_ids):
        n = len(report_ids)
        tr_n = math.ceil(n * self.train_perc)
        va_n = math.ceil(n * self.valid_perc)
        te_n = len(report_ids) - tr_n - va_n
        return report_ids[:tr_n], report_ids[tr_n:tr_n + va_n], report_ids[tr_n + va_n:]



# Groups + stratified (normal + remaining)
class BalancedKFold:
    def __init__(self, reports, images, n_folders=1, random_seed=datetime.now(), folder_val_perc=0.2):
        self.debug = True
        self.reports = reports
        self.n_reports = len(reports)
        self.images = images
        self.n_folders = n_folders
        self.random_seed = random_seed
        self.validation_perc = folder_val_perc

        # numbers for splitting examples
        self.distribution = 0.0
        self.folder_size = 0
        self.n_neg_per_fld = 0
        self.n_pos_per_fld = 0
        self.neg_indexes = None
        self.pos_indexes = None

        # indexes of examples len(dataset) % n_folders, i.e. examples that will redistributed
        self.additional_pos_idxs = None
        self.additional_neg_idxs = None

        self.n_neg = 0
        self.n_pos = 0

        self.partition_classes()

    def get_name(self):
        return 'cross_validation'

    def get_n_splits(self):
        return self.n_folders

    def partition_classes(self):
        r = self.reports

        iii = r.major_mesh == 'normal'
        self.distribution = np.count_nonzero(iii) / len(r)
        if self.debug:
            print('Percentage of NEGATIVE instances:', self.distribution)
            print(f"Total reports: {self.n_reports}, negative instances: {np.count_nonzero(iii)}")

        neg_indexes = r.loc[iii, 'id'].to_numpy(dtype=int) # r.index[iii].to_numpy(dtype=int)
        pos_indexes = r.loc[~iii, 'id'].to_numpy(dtype=int)  # r.index[~iii].to_numpy(dtype=int)
        # neg_indexes = r.index[iii].to_numpy()
        # pos_indexes = r.index[~iii].to_numpy()

        # shuffling here for keeping self.reports ordered
        random.shuffle(neg_indexes)
        random.shuffle(pos_indexes)
        self.neg_indexes = neg_indexes
        self.pos_indexes = pos_indexes

        self.n_neg = len(neg_indexes)
        self.n_pos = len(pos_indexes)
        if self.debug:
            print("|neg|= {}, |pos|= {}".format(self.n_neg, self.n_pos))

        # use math.floor here to use the two 'additional' lists below,
        # otherwise use math.ceil
        self.n_neg_per_fld = math.ceil(self.n_neg / self.n_folders)
        self.n_pos_per_fld = math.ceil(self.n_pos / self.n_folders)

        max_neg_idx = self.n_neg_per_fld * self.n_folders
        max_pos_idx = self.n_pos_per_fld * self.n_folders
        self.additional_neg_idxs = self.neg_indexes[max_neg_idx:]
        self.additional_pos_idxs = self.pos_indexes[max_pos_idx:]
        if self.debug:
            print('Additional negs: ', len(self.additional_neg_idxs))
            print('Additional poss: ', len(self.additional_pos_idxs))

        self.neg_indexes = self.neg_indexes[0:max_neg_idx]
        self.pos_indexes = self.pos_indexes[0:max_pos_idx]

        self.folder_size = self.n_neg_per_fld + self.n_pos_per_fld
        if self.debug:
            print("|neg| in folder:", self.n_neg_per_fld)
            print("|pos| in folder:", self.n_pos_per_fld)
            print("|folder|=", self.folder_size)
            print(f"Total |examples|= {self.n_reports}, |folder|*n_folders= {self.folder_size * self.n_folders}")

    def split(self, split_index):
        test_neg, test_pos = self.indexes_for_split(split_index)
        test_set = np.concatenate((test_neg, test_pos), axis=None)

        train_neg = [i for i in self.neg_indexes if i not in test_neg]
        train_pos = [i for i in self.pos_indexes if i not in test_pos]

        # train - validation
        n_neg_val = math.floor(len(train_neg) * self.validation_perc)
        n_pos_val = math.floor(len(train_pos) * self.validation_perc)

        val_neg = train_neg[0:n_neg_val]
        val_pos = train_pos[0:n_pos_val]
        train_neg = train_neg[n_neg_val:]
        train_pos = train_pos[n_pos_val:]

        training_set = np.concatenate((train_neg, train_pos), axis=None)
        validation_set = np.concatenate((val_neg, val_pos), axis=None)

        # distribute additional examples
        if split_index < len(self.additional_pos_idxs):
            z = np.zeros_like(self.additional_pos_idxs, dtype=bool)
            z[split_index] = True
            test_set = np.append(test_set, self.additional_pos_idxs[z])
            training_set = np.append(training_set, self.additional_pos_idxs[~z])
        if split_index < len(self.additional_neg_idxs):
            z = np.zeros_like(self.additional_neg_idxs, dtype=bool)
            z[split_index] = True
            test_set = np.append(test_set, self.additional_neg_idxs[z])
            training_set = np.append(training_set, self.additional_neg_idxs[~z])

        random.shuffle(training_set)
        random.shuffle(validation_set)
        # useless, but doable if needed
        # random.shuffle(test_set)
        if self.debug:
            print('Training set')
            self.check_set(training_set)
            print('Validation set')
            self.check_set(validation_set)
            print('Test set')
            self.check_set(test_set)
        return training_set, validation_set, test_set

    def indexes_for_split(self, split_index):
        neg_l = self.n_neg_per_fld * split_index
        neg_r = min(neg_l + self.n_neg_per_fld, self.n_neg)
        pos_l = self.n_pos_per_fld * split_index
        pos_r = min(pos_l + self.n_pos_per_fld, self.n_pos)

        res_neg = self.neg_indexes[neg_l:neg_r]
        res_pos = self.pos_indexes[pos_l:pos_r]
        if self.debug:
            print("Folder", split_index)
            print("Returning |negative|", len(res_neg))
            print("Returning |positive|", len(res_pos))
            print("Returning |total|", len(res_neg) + len(res_pos))
        return res_neg, res_pos

    def get_random_seed(self, seed):
        return self.seed

    def partition(self, normal_class):
        pass

    def check_set(self, set):
        subdf = self.reports[self.reports.id.isin(set)]
        iii = subdf.major_mesh == 'normal'
        d = np.count_nonzero(iii) / len(subdf)
        print('Distribution in current set: {:.5f}, original: {:.5f}'.format(d, self.distribution))



class RandomSplit:
    def __init__(self, reports, images, train_perc, valid_perc, random_seed=datetime.now()):
        self.csvr = reports
        self.csvi = images
        self.random_seed = random_seed
        random.seed(random_seed)

        self.n_reports = len(reports)
        self.n_images = len(images)
        self.report_ids = self.csvr['id'].to_numpy(dtype=int)
        self.train_ids, self.val_ids, self.test_ids = None, None, None
        self.img_training_set, self.img_test_set, self.img_validation_set = None, None, None
        self.train_perc = train_perc
        self.valid_perc = valid_perc

    def get_name(self):
        return 'random_splitter'

    def get_n_splits(self):
        return 1

    def split(self, split_index=0):
        test_perc = 1 - self.train_perc - self.valid_perc
        random.shuffle(self.report_ids)
        n_reps = len(self.report_ids)
        n_test = math.ceil(n_reps * test_perc)
        n_val = math.floor(n_reps * self.valid_perc)
        n_train = n_reps - n_val - n_test
        assert (n_train + n_val + n_test == n_reps)

        self.train_ids = self.report_ids[:n_train]
        self.val_ids = self.report_ids[n_train:n_train + n_val]
        self.test_ids = self.report_ids[n_train + n_val:]

        log('Actual |training|: %d' % len(self.train_ids))
        log('Actual |Validation|: %d' % len(self.val_ids))
        log('Actual |Test|: %d' % len(self.test_ids))

        # self.img_training_set = self.prepare_image_set(self.train_ids)
        # self.img_validation_set = self.prepare_image_set(self.val_ids)
        # self.img_test_set = self.prepare_image_set(self.test_ids)
        # return self.img_training_set, self.img_validation_set, self.img_test_set
        return self.train_ids, self.val_ids, self.test_ids

    # # use this method to prepare csv files containing image names, ids and encodings
    # def prepare_image_set(self, report_ids):
    #     # line in image_df:
    #     #   index, report_id, image_filename, encoded MeSH
    #     # i) select reports
    #     iii = self.csvi['report'].apply(lambda x: x in report_ids)
    #     # ii) select all columns but the first two: index, report_id
    #     cols = self.csvi.columns[2:]
    #     selected = self.csvi.loc[self.csvi.index[iii], cols]
    #     return selected


# ---
if __name__ == "__main__":
    reps = pd.read_csv("c:/Users/cardillo/data/deephealth/std-dataset/tsv/reports.tsv", sep=uc5def.csv_separator,
                       na_filter=False)
    imgs = pd.read_csv('c:/Users/cardillo/data/deephealth/std-dataset/tsv/encoded/e_image_labels.tsv', sep=uc5def.csv_separator,
                       na_filter=False)
    rnd_s = 40
    nf = 10
    # p = RandomSplit(reps, imgs, 0.7, 0.1, random_seed=1)
    # p = BalancedSplit(reps, imgs, 0.7, 0.1, random_seed=1)
    p = BalancedKFold(reps, None, n_folders=nf, random_seed=rnd_s)
    t_s, v_s, te_s = p.split(0)

    for j in range(nf):
        t, v, e = p.split(j)
        total = len(t) + len(v) + len(e)
        print('Folder {}, training {}, val {}, test {}, total examples in split {}'.format(j, len(t), len(v), len(e),
                                                                                           total))
