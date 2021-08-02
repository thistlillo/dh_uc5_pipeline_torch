"""This script cleans the texts in the tsv file produced by step A00. 

This is the second step (01) in pipeline A.


* PLEASE NOTICE:
* UC5 pipelines are managed via a Makefile. A simple modification of a file will force make to re-run the
* entire pipeline. Remember to use "touch" to avoid unnecessary runs:
*   touch -r <oldfile> <script> (in this case, the current .py file)

Author: Franco Alberto Cardillo <francoalberto.cardillo@ilc.cnr.it>
Date: July 2021
"""

import fire
import numpy as np
import os
import pandas as pd
from posixpath import join

import common.defaults as uc5def
import common.fileutils as fu
import text.cleaning as clean


class Cleaner:
    def __init__(self, in_img_fld, in_tsv_fld, out_fld, logger=None, log_level="info"):
        self.version = 3.0
        self.description = "DeepHealth UC5 raw dataset cleaner"
        
        if logger is None:
            logger =  uc5def.get_logger("cleaner", log_level)
        self.logger = logger
        
        self.img_folder = in_img_fld  # conf.in_folder(uc5keys.images)
        self.tsv_folder = in_tsv_fld
        
        # files to be cleaned are in out_folder
        self.out_folder = out_fld  # conf.in_folder(uc5keys.csvs)
        
        self.initial_check()
        with open(join(self.tsv_folder, uc5def.out_file_reports_raw)) as f:
            self.csvr = pd.read_csv(f, sep=uc5def.csv_separator, na_filter=False)
            self.logger.info("reports read, number of lines: %d" % len(self.csvr.index))
        
        # self.tmp_csvi = None  # holds temporary results for image csv
        self.tmp_csvr = None  # holds remporary results for report csv


    def initial_check(self):
        logger = self.logger
        
        logger.info("-")
        logger.info("Cleaner.py [v%.1f], %s" % (self.version, self.description))
        logger.info("image folder %s" % self.img_folder)
        logger.info("output folder %s" % self.out_folder)

        in_folders = [self.img_folder, self.tsv_folder]
        out_folders = [self.out_folder]

        ok = fu.check_folders_procedure(in_folders=in_folders, out_folders=out_folders,
                                        exist_ok=True, log_f=self.logger.info)
        if not ok:
            error_code = 1
            logger.error("Error from fileutils.check_folders_procedure. Exiting with error code %d", error_code)
            logger.error("-")
            exit(error_code)

    def clean(self):
        # STEP remove reports without associated images
        self.tmp_csvr = self.remove_reports_without_images(self.csvr)
       
        # STEP fill empty 'impression' on NORMAL reports
        self.tmp_csvr = self.fill_empty_impression(self.tmp_csvr)
        
        # save the two verified files, without further processing
        filename = join(self.out_folder, uc5def.out_file_reports_ver)
        self.tmp_csvr.to_csv(filename, sep=uc5def.csv_separator)
        self.logger.info("Saved: %s" % filename)

        # verified files, lowercase
        self.tmp_csvr = self.all_to_lower_case(self.tmp_csvr)  # , self.tmp_csvi)
        # save verified files, with text in lower case
        filename = join(self.out_folder, uc5def.out_file_reports_lower)
        self.tmp_csvr.to_csv(filename, sep=uc5def.csv_separator)
        self.logger.info("Saved: %s" % filename)
        
        # STEP remove incomplete reports
        self.tmp_csvr = self.drop_incomplete_reports(self.tmp_csvr)
        self.tmp_csvr = self.clean_texts(self.tmp_csvr)
        self.csvr = self.tmp_csvr
             
    def drop_incomplete_reports(self, csvr):
        # remove reports with empty impression and findings
        iii = csvr['impression'].str.len() == 0
        jjj = csvr['findings'].str.len() == 0
        kkk = iii & jjj
        self.logger.info("Dropping %d incomplete reports" % np.count_nonzero(kkk))
        csvr.drop(csvr.index[kkk], inplace=True)
        return csvr

    def remove_reports_without_images(self, csvr):
        iii = (csvr['n_images'] == 0)
        n_without_images = np.count_nonzero(iii)
        csvr = csvr.drop(csvr[iii].index)
        self.logger.info("Number of reports without images: %d" % n_without_images)
        return csvr

    def fill_empty_impression(self, csvr):
        iii = csvr.impression.str.len() == 0
        jjj = csvr.major_mesh.str.lower() == 'normal'
        kkk = iii & jjj
        csvr.loc[csvr.index[kkk], 'impression'] = 'normal'
        n_filled_impression = np.count_nonzero(kkk)
        self.logger.info("Filling %d empty 'impression' with 'normal'" % n_filled_impression)
        return csvr

    def all_to_lower_case(self, csvr):  # , csvi):
        # i_text_cols = ['image_caption', 'major_mesh', 'findings', 'impression']
        r_text_cols = ['major_mesh', "major_mesh", "findings", "impression", "indication"]
        type2cols = {'reports': r_text_cols}  #, 'images': i_text_cols}
        type2csv = {'reports': csvr}  # , 'images': csvi}
        assert (type2cols.keys() == type2csv.keys())

        for k, cols in type2cols.items():
            csv = type2csv[k]
            for col in cols:
                self.logger.debug('csv: %s, to lower case column: %s' % (k, col))
                csv[col] = csv[col].map(lambda x: ' '.join(s.lower() for s in x.split())) # lowercase and remove double or more spaces between words
        return csvr

    def clean_texts(self, csvr):  # , csvi):
        # csvr is currently unused here, passed in case we decide to clean also the major_mesh column
        func = lambda x: clean.minimal_cleaning(x)
        csvr.impression = csvr.impression.map(func)
        csvr.findings = csvr.findings.map(func)
        csvr.indication = csvr.indication.map(func)
        return csvr


    def simplify_mesh(self):
        func = lambda x: uc5def.seq_separator.join([y.split("/")[0] for y in x.split(uc5def.seq_separator)])
        csvr = self.csvr
        csvr['major_mesh'] = csvr['major_mesh'].map(func)

    def save(self):
        filename = os.path.join(self.out_folder, uc5def.out_file_reports)
        self.csvr.to_csv(filename, sep=uc5def.csv_separator)
        self.logger.info("Saved cleaned file: %s" % filename)


def  main(in_img_fld, in_tsv_fld, out_fld, log_level="info"):
    """Script for producing raw tsv files for the dataset. These files need to be further processed by subsequent steps in the A pipeline.
    
    Args:
        in_img_fld (string): input folder containing the images.
        in_tsv_fld (string): input folder containing the raw tsv files.
        out_fld (string): output folder for the intermediate and the final tsv files.
        log_level (string): log level as defined in python logging module (without the "logging." prefix). Default, info.
    """
    
    logger = uc5def.get_logger(__file__, log_level)
    
    cleaner = Cleaner(in_img_fld, in_tsv_fld, out_fld, logger=logger)
    cleaner.clean()
    cleaner.simplify_mesh()
    cleaner.save()

if __name__ == "__main__":
    fire.Fire(main)
