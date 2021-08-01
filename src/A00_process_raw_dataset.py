"""Process UC5 default dataset and produces tsv files, that need further processing before training.

This is the first step (00) in pipeline A.

This script reads the XML reports in the input folder and extracts relevant fields. For each reports,
the script checks the presence of the associated images in the image folder and removes report whose images
are absent from the output tsv files.

* PLEASE NOTICE:
* UC5 pipelines are managed via a Makefile. A simple modification of a file will force make to re-run the
* entire pipeline. Remember to use "touch" to avoid unnecessary runs:
*   touch -r <oldfile> A00_process_raw_dataset.py

Author: Franco Alberto Cardillo <francoalberto.cardillo@ilc.cnr.it>
Date: July 2021
"""

from bs4 import BeautifulSoup
import codecs
import fire
import humanize
import logging
import os
import os.path
import posixpath
import pandas as pd
import sys
from tqdm import tqdm

import common.defaults as uc5def
import common.fileutils as fu
import common.miscutils as mu
import text.report as report


# global variables
description = 'Process standard UC5 raw dataset'
version = 3.0

img_folder = None
txt_folder = None
out_folder = None
logger = None
verbose_log = None

out_file_reports = uc5def.out_file_reports_raw  # "reports_raw" + uc5def.csv_extension
out_file_images = uc5def.out_file_images_raw  # "images_raw" + uc5def.csv_extension


def parse_single_report(fn):
    global logger
    xml = fu.read_utf8_content(fn)
    
    soup = BeautifulSoup(xml, "lxml")
    d = report.parse_soup(soup)

    if verbose_log and logger.isEnabledFor(logging.DEBUG):
        logger.debug("File %s parsed, results:" % fn)
        for k in d.keys():
            logger.debug("Key: %s: %s" % (k, d[k]))
            if k == report.k_images:
                for le in d[k]:
                    logger.debug(le)
    return d


def parse_reports(folder, ext='xml'):
    global logger, verbose_log
    count = 0
    file_list = sorted(os.listdir(folder))
    n_files = len(file_list)
    logger.info("%d reports to parse" % n_files)
    results = {}
    counter = 0
    with tqdm(total=n_files) as pbar:
        for fn in file_list:
            p = posixpath.join(folder, fn)
            if not (os.path.isfile(p) and fn.endswith(ext)):
                logger.debug("Skipping %s" % fn)
                continue
            count = count + 1
            if verbose_log:
                logger.debug("Reading and parsing (as report) file: %s", fn)
            parsed = parse_single_report(p)
            results[fn] = parsed
            # TODO: do not update at each iteration
            # TODO: check whether or not tqdm is installed
            counter = counter + 1
            pbar.update(1)

    logger.info('Number of Parsed reports: %d  (%s) (memory: %s)' % (count, folder, humanize.naturalsize(sys.getsizeof(results))))

    return results


def build_report_header():
    # see build_report_line for the correct order of column names
    global logger
    header = [  # 'lineno',
              'filename',
              report.k_identifier,
              'n_major_mesh',
              report.k_major_mesh,
              'n_' + report.k_auto_term,
              report.k_auto_term,
              'n_images',
              report.k_image_filename,
              'indication',
              'findings',
              'impression',
              'version'
              ]
    header = uc5def.csv_separator.join(header)
    logger.debug("csv header for reports: %s" % header)
    return header


def build_report_line(filename, parsed_dict):
    # if not hasattr(build_report_line, "lineno"):
    #     build_report_line.lineno = 0  # it doesn't exist yet, so initialize it

    strings = [filename]  # [str(build_report_line.lineno), filename]
    p = parsed_dict  # just a shortcut

    strings.append(p[report.k_identifier])
    strings.append(str(len(p[report.k_major_mesh])))
    strings.append(uc5def.seq_separator.join(p[report.k_major_mesh]))
    strings.append(str(len(p[report.k_auto_term])))
    strings.append(uc5def.seq_separator.join(p[report.k_auto_term]))
    img_filenames = [d[report.k_image_filename] for d in p[report.k_images]]
    strings.append(str(len(img_filenames)))
    strings.append(uc5def.seq_separator.join(img_filenames))
    strings.append(p[report.k_indication])
    strings.append(p[report.k_findings])
    strings.append(p[report.k_impression])
    strings.append(str(version))
    # build_report_line.lineno += 1
    return uc5def.csv_separator.join(strings)


def build_image_header():
    # see build_image_line for the correct order of column names
    global logger
    header = ['lineno', report.k_image_filename,
              report.k_image_caption,
              'n_major_mesh',
              report.k_major_mesh,
              report.k_findings,
              report.k_impression,
              'report']
    header = uc5def.csv_separator.join(header)
    logger.debug("csv header for images: %s" % header)
    return header


def build_image_line(filename, parsed_dict):
    if not hasattr(build_image_line, "lineno"):
        build_image_line.lineno = 0  # it doesn't exist yet, so initialize it

    # parsed_dict is a list of dictionaries, one per image
    n_major_mesh = str(len(parsed_dict[report.k_major_mesh]))
    major_mesh = uc5def.seq_separator.join(parsed_dict[report.k_major_mesh])
    img_l = parsed_dict[report.k_images]
    lines = []

    for img_d in img_l:
        line = [str(build_image_line.lineno), img_d[report.k_image_filename],
                img_d[report.k_image_caption],
                n_major_mesh,
                major_mesh,
                # parsed_dict[report.k_indication],
                parsed_dict[report.k_findings],
                parsed_dict[report.k_impression], filename]
        newline = uc5def.csv_separator.join(line)
        lines.append(newline)
        build_image_line.lineno += 1
    return "\n".join(lines)


def save_results(parsed, output_fld):
    # Two files will be saved
    # 1) reports.csv, with header:
    #    version, report_id, report_filename, mesh_major, mesh_minor, n_images, [images]+
    # 2) images.csv, with header:
    #    version, report_id, report_filename, image_filename, mesh_major, mesh_minor, ...
    #    caption, indication, findings, impressions
    # NOTICE: in lists, such as mesh_major, the comma ',' will be substitued with '|'
    global logger
    
    if not os.path.exists(output_fld):
        logger.info(f"Creating output folder {output_fld}")
        os.makedirs(output_fld, exist_ok=True)
        
    rep_fn = posixpath.join(output_fld, out_file_reports)
    img_fn = posixpath.join(output_fld, out_file_images)

    logger.info("About to save files: %s, %s" % (rep_fn, img_fn))
    n_without = 0  # number of reports without images
    n_reports = len(parsed.keys())
    with codecs.open(rep_fn, "w", "utf-8") as rep_f, codecs.open(img_fn, "w", "utf-8") as img_f:
        rep_f.write(build_report_header() + "\n")
        img_f.write(build_image_header() + "\n")

        with tqdm(total=n_reports) as pbar:
            for filename, values in parsed.items():
                rep_line = build_report_line(filename, values)
                rep_f.write(rep_line + "\n")

                img_lines = build_image_line(filename, values)
                if len(img_lines.strip()) > 0:
                    img_f.write(img_lines + "\n")
                else:
                    logger.warning("WARNING! report %s without images", filename)
                    n_without = n_without + 1
                pbar.update(1)

    perc = round(float(n_without) / n_reports * 100, 2)
    logger.info("Number of reports without images: %d (%.2f%%)" % (n_without, perc))


def main(in_txt_fld, in_img_fld, out_fld, log_level="info", verbose=False):
    """Script for producing raw tsv files for the dataset. These files need to be further processed by subsequent steps in the A pipeline.
    
    Args:
        in_txt_fld (string): input folder containing the reports in xml format.
        in_img_fld (string): input folder containing the images associated to the reports (expected PNG format).
        out_fld (string): output folder for the tsv files.
        log_level (string): log level as defined in python logging module (without the "logging." prefix). Default, info.
        verbose (boolean): log many more messages, showing, for example, the parsing of individual xml files. Default, False.
    """
    global conf, img_folder, txt_folder, out_folder, logger, verbose_log

    logger = uc5def.get_logger(__file__, log_level)
    verbose_log = verbose
    
    img_folder = in_img_fld  # conf.in_folder(uc5keys.images)
    txt_folder = in_txt_fld  # conf.in_folder(uc5keys.texts)
    out_folder = out_fld  # conf.in_folder(uc5keys.csvs)
    
    logger.info("-")
    logger.info("process_raw_dataset [v%.1f], %s" % (version, description))
    logger.info("image folder %s" % img_folder)
    logger.info("text folder %s" % txt_folder)
    logger.info("output folder %s" % out_folder)
    
    in_folders = [img_folder, txt_folder]
    out_folders = []  # [out_folder]

    ok = fu.check_folders_procedure(in_folders=in_folders, out_folders=out_folders,
                                    exist_ok=True, log_f=logger.warning)
    if not ok:
        logger.error("Exiting, error code 1")
        logger.error("-")
        exit(1)

    parsed = parse_reports(txt_folder)
    save_results(parsed, out_folder)

    logger.info("all went well")


if __name__ == "__main__":
    fire.Fire(main)
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--in_img_fld", type=str, help="folder with images")
    # parser.add_argument("--in_txt_fld", type=str, help="folder with text reports")
    # parser.add_argument("--out_fld", type=str, help="output folder for tsv files")
    # main(parser.parse_args(sys.argv[1:]))