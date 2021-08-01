import gensim
import humanize
import os
from common.miscutils import Timer

# we are not currently using word embeddings
embeddings_fld = "c:/users/cardillo/data/deephealth/embeddings"
embeddings_fn = "bio-nlp/PubMed-w2v.bin"

def load_keyed_vectors():
    path = os.path.join(embeddings_fld, embeddings_fn)
    is_binary=True
    kv_fn = "embeddings/bio-nlp/kv_untrainable_fastload.bin"
    if not os.path.exists(kv_fn):
        with Timer(text="Time for extracting keyed-vectors: {:.1f} secs"):
            print("Loading embeddings:", path)
            keyvects = gensim.models.KeyedVectors.load_word2vec_format(
                os.path.join(embeddings_fld, embeddings_fn),
                binary=is_binary)
            keyvects.save(kv_fn)
            print("Saved:", kv_fn)
    else:
        with Timer(text="Time for loading keyed-vectors: {:.1f} secs"):
            print("Using pre-extracted keyed-vectors in:", kv_fn)
            mmap = "r"
            keyvects = gensim.models.KeyedVectors.load(kv_fn, mmap="r")
    return keyvects

def filter_embeddings(kv):
    d = kv.vocab.keys()
    v = kv.vectors
    print(f"Got vocabulary {type(kv.vocab)}, |keys|: {len(d)}")
    print(f"Got vectors {type(v)}, shape: {v.shape}")

    return 0

def main():
    # load embeddings
    kv = load_keyed_vectors()
    kv = filter_embeddings(kv)

if __name__ == "__main__":
    main()