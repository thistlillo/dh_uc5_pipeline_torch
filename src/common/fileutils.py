import codecs
import os
import pandas as pd

import common.defaults as uc5def

def check_existing_folder(fn):
    return os.path.exists(fn) and os.access(fn, os.R_OK)


def create_writable_folder(fn, exist_ok=False):
    ok = True
    msg = ''

    try:
        os.makedirs(fn, exist_ok=exist_ok)
        # if exist_ok, then check we can write
        if exist_ok and not os.access(fn, os.W_OK):
            ok = False
            msg = 'Folder exists, but it is not writable'
    except OSError as exc:
        ok = False
        msg = str(exc)

    return ok, msg


def check_folders(r, w, exist_ok=False):
    read_flags = []

    for fn in r:
        read_flags.append(check_existing_folder(fn))

    if not all(read_flags):
        return read_flags, ([], [])

    write_flags = []
    write_msgs = []
    for fn in w:
        f, m = create_writable_folder(fn, exist_ok=exist_ok)
        write_flags.append(f)
        write_msgs.append(m)

    return read_flags, (write_flags, write_msgs)


def check_folders_procedure(in_folders, out_folders, exist_ok, log_f=None):
    if log_f is None:
        log_f = print
    r_flags, (w_flags, w_msgs) = check_folders(r=in_folders, w=out_folders,
                                               exist_ok=exist_ok)

    # check 'in' folders
    in_ok = all(r_flags)
    out_ok = True
    if not in_ok:
        for i, flag in enumerate(r_flags):
            if not flag:
                log_f('Folder %s does not exist or is not readable' % in_folders[i])
        log_f("\t Output folder(s) ignored. No changes to filesystem have been made.")
        return False

    out_ok = all(w_flags)
    if not out_ok:
        prev_folders = None
        for i, flag in enumerate(w_flags):
            if not flag:
                log_f("Problem with %s: %s" % (out_folders[i], w_msgs[i]))
                if i > 0:
                    log_f('\t--> filesystem might have changed. Check the following folder(s):\n\t%s' % prev_folders)
            else:
                prev_folders = out_folders[i] if prev_folders is None else prev_folders + "\n\t" + out_folders[i]

    return out_ok


def filename_from_path(path, keep_extension=True):
    base = os.path.basename(path)
    if keep_extension:
        return base

    pre, ext = os.path.splitext(base)
    return pre


def change_extension(filename, new_ext):
    pre, ext = os.path.splitext(filename)
    return pre + "." + new_ext


def read_csv(filename, sep=uc5def.csv_separator, nafilter=False):
    if (not filename.endswith(".csv")) and (not filename.endswith(".tsv")):
        filename = filename + uc5def.csv_extension
    return pd.read_csv(filename, sep=sep, na_filter=nafilter)


def read_utf8_content(filename):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return content


def save_as_utf8(text, abs_path):
    file = codecs.open(abs_path, "w", "utf-8")
    file.write(text)
    file.close()