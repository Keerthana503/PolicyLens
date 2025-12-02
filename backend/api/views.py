# backend/api/views.py
import os
import json
import pickle
import re
import unicodedata
import logging
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from django.conf import settings
from django.contrib.auth.models import User
from rest_framework import generics, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response

from .serializers import RegisterSerializer, UserSerializer, DocumentSerializer
from .models import Document

logger = logging.getLogger(__name__)

from api.utils.tfidf_utils import validate_artifacts

# -------------------------
# Sentence-Transformer (cached) for semantic re-ranking
# -------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except Exception:
    SentenceTransformer = None
    util = None
    torch = None

_SENTENCE_MODEL = None


def get_sentence_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Return a cached SentenceTransformer instance or None if not installed.
    """
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL
    if SentenceTransformer is None:
        return None
    # Load to CPU by default; move to GPU in your environment if desired.
    _SENTENCE_MODEL = SentenceTransformer(model_name)
    return _SENTENCE_MODEL


# -------------------------
# Auth views (register / me)
# -------------------------
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer


class MeView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


# -------------------------
# Documents list & upload
# -------------------------
class DocumentListCreateView(generics.ListCreateAPIView):
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user).order_by("-uploaded_at")

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user, status="uploaded")


# -------------------------
# TF-IDF query view (lightweight + optional re-rank + debug)
# -------------------------
class QueryTfIdfAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        data = request.data or {}
        # doc_id might come as int or string
        doc_id = data.get("doc_id") or request.query_params.get("doc_id")
        question = data.get("question") or request.query_params.get("question")
        # debug flag: JSON body field `debug` or query param `debug=1|true`
        debug_flag = data.get("debug", None) or request.query_params.get("debug", None)
        debug = False
        if debug_flag is not None:
            debug = str(debug_flag).lower() in ("1", "true", "yes", "y")

        try:
            top_k = int(data.get("top_k", request.query_params.get("top_k", 5)))
        except Exception:
            top_k = 5

        if not doc_id or not question:
            return Response({"detail": "doc_id and question are required"},
                            status=status.HTTP_400_BAD_REQUEST)

        # make doc_id int if possible
        try:
            doc_id = int(doc_id)
        except Exception:
            return Response({"detail": "doc_id must be an integer"}, status=status.HTTP_400_BAD_REQUEST)

        # verify document & ownership
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({"detail": "Document not found"}, status=status.HTTP_404_NOT_FOUND)
        if doc.owner_id != request.user.id:
            return Response({"detail": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

        # log query start
        logger.info(
            "TFIDF query start: doc=%s user=%s question=%s top_k=%s debug=%s",
            doc.id,
            request.user.id,
            (question[:140] + "...") if len(question) > 140 else question,
            top_k,
            debug,
        )

        # per-document artifact paths
        tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
        base = f"doc_{doc.id}"
        matrix_path = os.path.join(tfidf_dir, f"{base}_matrix.npz")
        vocab_json_path = os.path.join(tfidf_dir, f"{base}_vocab.json")
        vect_pkl_path = os.path.join(tfidf_dir, f"{base}_vectorizer.pkl")
        meta_path = os.path.join(tfidf_dir, f"{base}_metadata.json")

        # --- Artifact validation (new)
        from api.utils.tfidf_utils import validate_artifacts
        ok, reason, details = validate_artifacts(matrix_path, vocab_json_path, vect_pkl_path)
        if not ok:
        # if debug mode, include full details
            if debug:
                return Response({"detail": f"Artifact error: {reason}", "debug": details}, status=500)
            else:
                return Response({"detail": f"TF-IDF index invalid: {reason}"}, status=500)

        # require at least matrix + metadata
        if not (os.path.exists(matrix_path) and os.path.exists(meta_path)):
            logger.error("Missing artifacts for doc=%s (matrix or metadata)", doc.id)
            return Response({"detail": "TF-IDF index for this document is not available. Run ingestion."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # load matrix and metadata
        try:
            X = sparse.load_npz(matrix_path)
            with open(meta_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
        except Exception as e:
            logger.exception("Failed to load TF-IDF matrix/metadata for doc=%s", doc.id)
            return Response({"detail": f"Failed to load TF-IDF matrix/metadata: {e}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        debug_info = {
            "matrix_shape": None,
            "vocab_size": None,
            "vocab_from_pickle": False,
            "q_vec_shape": None,
            "top_tfidf_candidates": [],
        }
        debug_info["matrix_shape"] = tuple(X.shape)
        logger.info("Loaded TF-IDF artifacts for doc=%s: matrix_shape=%s metadata_chunks=%d",
                    doc.id, tuple(X.shape), len(metadata))

        # -------------------------
        # Load or construct vectorizer (prefer pickled fitted vectorizer)
        # -------------------------
        vectorizer = None

        # 1) Try pickled fitted vectorizer first (preferred)
        if os.path.exists(vect_pkl_path):
            try:
                with open(vect_pkl_path, "rb") as vf:
                    candidate = pickle.load(vf)
                # attempt to find vocabulary size from candidate safely
                cand_vocab_size = None
                if hasattr(candidate, "vocabulary_"):
                    cand_vocab_size = len(candidate.vocabulary_)
                elif hasattr(candidate, "vocabulary"):
                    cand_vocab_size = len(candidate.vocabulary)
                # ensure dims match: only accept pickle if feature count == matrix width
                if cand_vocab_size is not None and cand_vocab_size == X.shape[1]:
                    vectorizer = candidate
                    debug_info["vocab_from_pickle"] = True
                    debug_info["vocab_size"] = cand_vocab_size
                    logger.info("Loaded pickled vectorizer for doc=%s vocab_size=%s", doc.id, cand_vocab_size)
                else:
                    # incompatible pickle -> ignore (safe fallback)
                    vectorizer = None
                    debug_info["vocab_size"] = cand_vocab_size
                    logger.warning("Ignored pickled vectorizer for doc=%s: pickled_vocab=%s matrix_features=%s",
                                   doc.id, cand_vocab_size, X.shape[1])
            except Exception as e:
                vectorizer = None
                debug_info["vocab_load_error"] = str(e)
                logger.exception("Error loading pickled vectorizer for doc=%s", doc.id)

        # 2) Fallback: build vectorizer from vocab.json + fit idf_ on stored chunks
        if vectorizer is None:
            if os.path.exists(vocab_json_path):
                try:
                    with open(vocab_json_path, "r", encoding="utf-8") as vf:
                        vocab = json.load(vf)
                    # ensure indices are ints
                    vocab = {k: int(v) for k, v in vocab.items()}
                    debug_info["vocab_size"] = len(vocab)
                    vectorizer = TfidfVectorizer(stop_words="english", vocabulary=vocab)
                    texts = [m.get("text", "") for m in metadata]
                    # fit to produce idf_ (fast for small number of chunks)
                    vectorizer.fit(texts)
                    # sanity: verify feature count matches matrix
                    if vectorizer.transform(["test"]).shape[1] != X.shape[1]:
                        logger.error("Dimension mismatch after building vectorizer from vocab.json for doc=%s", doc.id)
                        return Response(
                            {"detail": "Dimension mismatch after building vectorizer from vocab.json. Re-ingest document."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        )
                    logger.info("Built vectorizer from vocab.json for doc=%s vocab_size=%s", doc.id, len(vocab))
                except Exception as e:
                    logger.exception("Failed to build vectorizer from vocab.json for doc=%s", doc.id)
                    return Response({"detail": f"Failed to prepare TF-IDF vectorizer from vocab.json: {e}"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                logger.error("No pickled vectorizer and no vocab.json for doc=%s", doc.id)
                return Response(
                    {"detail": "Pickled vectorizer not found and no vocab.json fallback. Please re-run ingestion for this document."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        # --- sanity check: ensure feature dims match the saved matrix X
        try:
            q_vec = vectorizer.transform([question])   # shape (1, n_features)
            debug_info["q_vec_shape"] = tuple(q_vec.shape)
            logger.debug("Question vectorized for doc=%s q_vec_shape=%s", doc.id, tuple(q_vec.shape))
        except Exception as e:
            logger.exception("Failed to vectorize question for doc=%s", doc.id)
            return Response({"detail": f"Failed to vectorize question: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if q_vec.shape[1] != X.shape[1]:
            logger.error("Dimension mismatch for doc=%s: q_vec_features=%s matrix_features=%s",
                         doc.id, q_vec.shape[1], X.shape[1])
            return Response(
                {"detail": f"Dimension mismatch: question vector has {q_vec.shape[1]} features but TF-IDF matrix has {X.shape[1]} features. Re-ingest the document to regenerate matching artifacts."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # safe to compute scores now
        try:
            scores = (q_vec.dot(X.T)).toarray()[0]
        except Exception as e:
            logger.exception("Failed to compute TF-IDF scores for doc=%s", doc.id)
            return Response({"detail": f"Failed to compute scores: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        nonzero = int((scores > 0).sum())
        logger.info("Scored doc=%s question; nonzero_scores=%d total_chunks=%d", doc.id, nonzero, X.shape[0])

        ranked = np.argsort(-scores)

        # Gather top TF-IDF hits (take some extra for re-rank)
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx]
            if item.get("doc_id") == doc.id:
                hits.append({
                    "score": float(scores[idx]),
                    "page": item.get("page"),
                    "text": item.get("text")
                })
            if len(hits) >= top_k * 3:
                break

        debug_info["top_tfidf_candidates"] = [
            {"idx": int(r), "score": float(scores[r]),
             "text": (metadata[r]["text"] if r < len(metadata) else "")}
            for r in ranked[: min(50, ranked.shape[0])]
        ]

        # -------------------------
        # Semantic Re-ranker Stage (optional)
        # -------------------------
        if hits:
            model = get_sentence_model()
            if model is None or util is None:
                # no model available: trim to top_k and continue
                hits = hits[:top_k]
            else:
                try:
                    question_emb = model.encode(question, convert_to_tensor=True)
                    texts_to_rank = [h["text"] for h in hits]
                    text_embs = model.encode(texts_to_rank, convert_to_tensor=True)
                    sim_scores = util.cos_sim(question_emb, text_embs)[0]

                    if torch is not None:
                        sorted_idx = torch.argsort(sim_scores, descending=True)
                        re_ranked = []
                        for i in sorted_idx:
                            ii = int(i)
                            h = hits[ii]
                            try:
                                s = float(sim_scores[ii].item())
                            except Exception:
                                s = float(np.array(sim_scores[ii]))
                            h["semantic_score"] = s
                            re_ranked.append(h)
                        hits = re_ranked[:top_k]
                    else:
                        sim_np = np.array(sim_scores)
                        order = np.argsort(-sim_np)
                        re_ranked = []
                        for ii in order:
                            h = hits[int(ii)]
                            h["semantic_score"] = float(sim_np[ii])
                            re_ranked.append(h)
                        hits = re_ranked[:top_k]
                except Exception:
                    logger.exception("Semantic re-rank failed for doc=%s — falling back to TF-IDF hits", doc.id)
                    hits = hits[:top_k]

        # ------------------------
        # Generic answer builder (document-agnostic; avoids returning noise lines)
        # ------------------------
        def clean_text_for_display(s: str) -> str:
            if not s:
                return ""
            # normalize unicode forms
            s = unicodedata.normalize("NFKC", s)
            # remove (cid:123) style artifacts and control chars
            s = re.sub(r"\(cid:\d+\)", " ", s)
            s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+", " ", s)
            # minimal noise stripping: emails/urls/handles/phones
            s = re.sub(r"\S+@\S+\.\S+", " ", s)                      # email
            s = re.sub(r"https?://\S+|www\.\S+", " ", s)             # urls
            s = re.sub(r"(github\.com|linkedin\.com)\S*", " ", s, flags=re.I)
            s = re.sub(r"\b\d{6,}\b", " ", s)                        # long numbers (phone-ish)
            # join hyphenated line-breaks and normalize newlines to spaces
            s = re.sub(r"-\s*\n\s*", "", s)
            s = s.replace("\n", " ")
            s = re.sub(r"[^\w\s\.,:;'\-()\/#@&]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # If no hits, empty answer
        if not hits:
            answer = ""
        else:
            # build a set of query keywords (ignore very short words)
            q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]

            # collect sentence candidates with combined score
            candidates = []
            seen_texts = set()

            for hit_index, h in enumerate(hits):
                chunk_text = clean_text_for_display(h.get("text", "") or "")
                if not chunk_text:
                    continue

                tfidf_score = float(h.get("score", 0.0))
                semantic_score = float(h.get("semantic_score", 0.0)) if h.get("semantic_score") is not None else 0.0

                # split into sentences (also split on newline-like breaks)
                sentences = re.split(r'(?<=[\.\?\!\n])\s+', chunk_text)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent or len(sent) < 20:
                        continue

                    # basic filters for noisy lines
                    lower = sent.lower()
                    if any(token in lower for token in ("github.com", "linkedin.com", "http", "@", "skills:", "cv", "resume")):
                        continue

                    # dedupe
                    if sent in seen_texts:
                        continue
                    seen_texts.add(sent)

                    sl = sent.lower()
                    kw_count = sum(1 for q in q_words if q in sl)

                    # combine signals into a single score:
                    score = (kw_count * 20.0) + (semantic_score * 5.0) + (tfidf_score * 1.0)

                    candidates.append({
                        "score": score,
                        "kw_count": kw_count,
                        "tfidf_score": tfidf_score,
                        "semantic_score": semantic_score,
                        "text": sent,
                        "hit_index": hit_index,
                    })

            # safeguard: cap candidates before sorting
            if len(candidates) > 2000:
                candidates = sorted(candidates, key=lambda c: -c["score"])[:2000]

            candidates.sort(key=lambda c: (-c["score"], abs(len(c["text"]) - 140)))

            # take top unique sentences (up to 3) and prefer diversity across hits
            selected = []
            selected_hit_idxs = set()
            for c in candidates:
                t = c["text"]
                hidx = c.get("hit_index")
                if hidx is not None and hidx not in selected_hit_idxs:
                    selected.append(t)
                    selected_hit_idxs.add(hidx)
                elif len(selected) < 3:
                    selected.append(t)
                if len(selected) >= 3:
                    break

            # fallback: if nothing selected, use first 200 chars of top hit
            if not selected:
                top_text = clean_text_for_display(hits[0].get("text", "")) if hits else ""
                snippet = top_text[:200].rstrip()
                answer = snippet + ("..." if len(top_text) > 200 else "")
            else:
                answer = " • ".join(selected)
                if len(answer) > 500:
                    answer = answer[:497].rstrip() + "..."

        resp = {"answer": answer, "hits": hits}
        if debug:
            # include some debug info: top candidates and shapes
            resp["debug"] = debug_info
            resp["debug"]["candidates_preview"] = [
                {"text": (h.get("text")[:300] + ("..." if len(h.get("text","")) > 300 else "")),
                 "score": h.get("score"), "semantic_score": h.get("semantic_score", None)}
                for h in hits[: min(50, len(hits))]
            ]

        logger.info("Returning answer for doc=%s user=%s answer_len=%d hits=%d debug=%s",
                    doc.id, request.user.id, len(answer or ""), len(hits), debug)
        if debug:
            logger.debug("Debug info for doc=%s: %s", doc.id, json.dumps(debug_info, default=str)[:2000])

        return Response(resp)