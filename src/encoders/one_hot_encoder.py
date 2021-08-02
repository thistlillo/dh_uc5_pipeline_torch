import numpy as np
import pandas as pd


class OneHotEncoder:
    ood = "_ood_"

    def __init__(self, term_list, add_out_of_dict=False, ignore_key_errors=False, datatype=int):
        self.add_out_of_dict = add_out_of_dict
        self.ignore_key_errors = ignore_key_errors
        self.datatype = datatype
        self.term_list = self.process_term_list(term_list)
        self.n_terms = len(self.term_list)
        self.enc_mat = None
        self.enc_df = None

    def process_term_list(self, term_list):
        term_list = sorted(list(set(term_list)))
        if self.add_out_of_dict:
            term_list = term_list + [self.ood]
        return term_list

    def get_dim(self):
        return self.n_terms

    def get_encoding_mat(self):
        if self.enc_mat is None:
            self.enc_mat = np.eye(self.n_terms, dtype=self.datatype)
        return self.enc_mat

    def get_encoding_df(self):
        if self.enc_df is None:
            df = pd.DataFrame(data=self.get_encoding_mat())
            df.insert(0, "term", self.term_list)
            df.set_index("term", inplace=True, verify_integrity=True)
            self.enc_df = df
        return self.enc_df

    def encode(self, term_l):
        # if self.add_out_of_dict:
        #     return self.encode_ood(term_l)
        # else:
        #     return self.encode_no_ood(term_l)
        return self.encode_ood(term_l)

    def encode_no_ood(self, term_l):
        e = self.get_encoding_df()
        rows = e.loc[term_l]
        return rows.any(axis=0).astype(self.datatype).to_numpy()

    def encode_ood(self, term_l):
        v = np.zeros(shape=(len(term_l), self.n_terms), dtype=self.datatype)
        e = self.get_encoding_df()
        for i, t in enumerate(term_l):
            try:
                v[i] = e.loc[t]
            except KeyError:
                if self.add_out_of_dict:
                    v[i] = e.loc[self.ood]
                elif not self.ignore_key_errors:
                    raise KeyError("out of dictionary: %s" % t)
        return v.any(axis=0).astype(self.datatype)

    def decode(self, v, include_ood=False):
        dec = self.get_encoding_df().index[v.astype(bool)].to_list()
        if (self.ood in dec) and (not include_ood):
            dec.remove(self.ood)
        return dec

    def get_terms(self, include_ood=False):
        out = self.term_list
        if self.add_out_of_dict and (not include_ood):
            out = out[:-1]
        return out
