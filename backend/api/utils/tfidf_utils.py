import json, os
from scipy import sparse
import pickle

def validate_artifacts(matrix_path, vocab_json_path=None, vect_pkl_path=None):
    """
    Returns (ok: bool, reason: str, details: dict)
    """
    details = {}
    if not os.path.exists(matrix_path):
        return False, "matrix_missing", details
    try:
        X = sparse.load_npz(matrix_path)
    except Exception as e:
        return False, f"matrix_load_error: {e}", details
    details['matrix_shape'] = X.shape

    if vect_pkl_path and os.path.exists(vect_pkl_path):
        try:
            with open(vect_pkl_path, "rb") as f:
                v = pickle.load(f)
            vocab_len = None
            if hasattr(v, "vocabulary_"):
                vocab_len = len(v.vocabulary_)
            elif hasattr(v, "vocabulary"):
                vocab_len = len(v.vocabulary)
            details['pickled_vocab_size'] = vocab_len
            if vocab_len is not None and vocab_len != X.shape[1]:
                return False, "pickled_vectorizer_dim_mismatch", details
        except Exception as e:
            return False, f"pickled_vectorizer_load_error: {e}", details

    if vocab_json_path and os.path.exists(vocab_json_path):
        try:
            v = json.load(open(vocab_json_path, "r", encoding="utf-8"))
            details['vocab_json_size'] = len(v)
            if len(v) != X.shape[1]:
                return False, "vocab_json_dim_mismatch", details
        except Exception as e:
            return False, f"vocab_json_load_error: {e}", details

    return True, "ok", details