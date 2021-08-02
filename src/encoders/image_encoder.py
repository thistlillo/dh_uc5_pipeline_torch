import numpy as np
import pandas as pd
from tqdm import tqdm

import common.defaults as uc5def
import common.miscutils as mu
from encoders.one_hot_encoder import OneHotEncoder


class ImageEncoder:
    def __init__(self, df, mesh_encoder, logger=None):
        self.df = df
        self.mesh_encoder = mesh_encoder
        self.encoding_df = None
        if logger is None:
            logger = uc5def.get_logger(__file__, "debug")
        self.logger = logger
        
    def get_encoding_df(self):
        if self.encoding_df is None:
            self.encoding_df = self.encode_images()
        return self.encoding_df

    def encode_images(self):
        m_enc = self.mesh_encoder
        if not ('major_mesh_l' in self.df.columns):
            mu.add_list_of_mesh(self.df)
        df = self.df
        n_rows = len(df)
        dim_enc = m_enc.get_dim()
        n_rows_enc = df['n_images'].sum()
        print("n images: %s" % n_rows_enc)
        encodings = np.zeros(shape=(n_rows_enc, dim_enc), dtype=int)
        reports = []
        filenames = []
        enc_row = 0
        for ri in tqdm(range(n_rows)):
            row = df.loc[ri]
            report_id = row['id']
            mesh_l = row['major_mesh_l']
            image_filenames = [x for x in row['image_filename'].split(uc5def.seq_separator)]
            enc = m_enc.encode(mesh_l)
            for img_fn in image_filenames:
                filenames.append(img_fn)
                reports.append(report_id)
                encodings[enc_row, :] = enc
                enc_row += 1

        out = pd.DataFrame(data=reports, columns=["reports"])
        out['filename'] = filenames
        encodings_df = pd.DataFrame(data=encodings, columns=m_enc.get_terms(include_ood=False))
        out = pd.concat([out, encodings_df], axis=1, ignore_index=True)
        out.columns = ["report", "filename"] + m_enc.get_terms(include_ood=False)
        return out

    def get_dim(self):
        return self.mesh_encoder.get_dim()
