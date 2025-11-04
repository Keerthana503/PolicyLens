# backend/api/views.py
import os
import json
import pickle
import re
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
# TF-IDF query view (lightweight + optional re-rank)
# -------------------------
class QueryTfIdfAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        data = request.data
        doc_id = data.get("doc_id")
        question = data.get("question")
        try:
            top_k = int(data.get("top_k", 5))
        except Exception:
            top_k = 5

        if not doc_id or not question:
            return Response({"detail": "doc_id and question are required"}, status=status.HTTP_400_BAD_REQUEST)

        # verify document & ownership
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({"detail": "Document not found"}, status=status.HTTP_404_NOT_FOUND)
        if doc.owner_id != request.user.id:
            return Response({"detail": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

        # per-document artifact paths
        tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
        base = f"doc_{doc.id}"
        matrix_path = os.path.join(tfidf_dir, f"{base}_matrix.npz")
        vocab_json_path = os.path.join(tfidf_dir, f"{base}_vocab.json")
        vect_pkl_path = os.path.join(tfidf_dir, f"{base}_vectorizer.pkl")
        meta_path = os.path.join(tfidf_dir, f"{base}_metadata.json")

        # require at least matrix + metadata
        if not (os.path.exists(matrix_path) and os.path.exists(meta_path)):
            return Response({"detail": "TF-IDF index for this document is not available. Run ingestion."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # load matrix and metadata
        try:
            X = sparse.load_npz(matrix_path)
            with open(meta_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
        except Exception as e:
            return Response({"detail": f"Failed to load TF-IDF matrix/metadata: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # -------------------------
        # Load or require a pickled, fitted vectorizer (important)
        # -------------------------
        # prefer a pickled fitted vectorizer (keeps idf_ and exact mapping)
        vectorizer = None
        if os.path.exists(vect_pkl_path):
            try:
                with open(vect_pkl_path, "rb") as vf:
                    candidate = pickle.load(vf)
                # sanity: ensure candidate has a vocabulary_ attribute and the feature size matches the matrix
                if hasattr(candidate, "vocabulary_"):
                    cand_size = len(candidate.vocabulary_)
                    if X.shape[1] == cand_size:
                        vectorizer = candidate
                    else:
                        # stale/incorrect pickle — ignore it
                        vectorizer = None
                else:
                    vectorizer = None
            except Exception:
                vectorizer = None
                
        # If no pickle, we still allow fallback to vocab.json but only if it's safe.
        if vectorizer is None:
            if os.path.exists(vocab_json_path):
                try:
                    with open(vocab_json_path, "r", encoding="utf-8") as vf:
                        vocab = json.load(vf)
                    # ensure indices are native ints
                    vocab = {k: int(v) for k, v in vocab.items()}
                    # build a vectorizer with exact vocabulary mapping
                    vectorizer = TfidfVectorizer(stop_words="english", vocabulary=vocab)
                    # set idf_ by fitting on stored chunks (fast)
                    texts = [m.get("text", "") for m in metadata]
                    vectorizer.fit(texts)
                except Exception as e:
                    return Response({"detail": f"Failed to prepare TF-IDF vectorizer from vocab.json: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                return Response(
                    {"detail": "Pickled vectorizer not found and no vocab.json fallback. Please re-run ingestion for this document."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        # --- sanity check: ensure feature dims match the saved matrix X
        try:
            q_vec = vectorizer.transform([question])   # shape (1, n_features)
        except Exception as e:
            return Response({"detail": f"Failed to vectorize question: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if q_vec.shape[1] != X.shape[1]:
            return Response(
                {"detail": f"Dimension mismatch: question vector has {q_vec.shape[1]} features but TF-IDF matrix has {X.shape[1]} features. Re-ingest the document to regenerate matching artifacts."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # safe to compute scores now
        try:
            scores = (q_vec.dot(X.T)).toarray()[0]
        except Exception as e:
            return Response({"detail": f"Failed to compute scores: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        ranked = np.argsort(-scores)

        # Gather top TF-IDF hits (take some extra for re-rank)
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            # safe guard index bounds in metadata
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
                    # encode question and candidate texts once
                    question_emb = model.encode(question, convert_to_tensor=True)
                    texts_to_rank = [h["text"] for h in hits]
                    text_embs = model.encode(texts_to_rank, convert_to_tensor=True)
                    sim_scores = util.cos_sim(question_emb, text_embs)[0]

                    # sort by semantic score descending and attach semantic_score
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
                    # on any failure during re-rank, fall back to TF-IDF hits
                    hits = hits[:top_k]

        # ------------------------
        # Generic answer builder (document-agnostic; no resume special-casing)
        # ------------------------
        def clean_text_for_display(s: str) -> str:
            if not s:
                return ""
            s = re.sub(r"\(cid:\d+\)", " ", s)
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # If no hits, empty answer
        if not hits:
            answer = ""
        else:
            # build a set of query keywords (ignore very short words)
            q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]

            # collect sentence candidates with scores
            candidates = []
            seen_texts = set()

            for h in hits:
                chunk_text = clean_text_for_display(h.get("text", "") or "")
                if not chunk_text:
                    continue

                # get chunk-level scores (if available)
                tfidf_score = float(h.get("score", 0.0))
                semantic_score = float(h.get("semantic_score", 0.0)) if h.get("semantic_score") is not None else 0.0

                # split into sentence-like pieces (works for most docs)
                sentences = re.split(r'(?<=[\.\?\!\n])\s+', chunk_text)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    # dedupe exact sentence text
                    if sent in seen_texts:
                        continue
                    seen_texts.add(sent)

                    # keyword match count (higher is better)
                    kw_count = sum(1 for q in q_words if q in sent.lower())

                    # basic sentence-length penalty: ignore extremely short meaningless pieces
                    if len(sent) < 20:
                        length_penalty = -1.0
                    else:
                        length_penalty = 0.0

                    # combine signals into a single score
                    # weights: keyword count strong, semantic moderate, tfidf small, length penalty
                    score = (kw_count * 10.0) + (semantic_score * 5.0) + (tfidf_score * 1.0) + length_penalty

                    candidates.append({
                        "score": score,
                        "kw_count": kw_count,
                        "tfidf_score": tfidf_score,
                        "semantic_score": semantic_score,
                        "text": sent,
                    })

            # avoid huge sorts on massive docs (optional safeguard)
            if len(candidates) > 2000:
                candidates = sorted(candidates, key=lambda c: -c["score"])[:2000]

            # sort candidates by combined score desc, then by length (prefer concise)
            candidates.sort(key=lambda c: (-c["score"], abs(len(c["text"]) - 140)))

            # take top unique sentences (up to 3)
            selected = []
            for c in candidates:
                t = c["text"]
                if t not in selected:
                    selected.append(t)
                if len(selected) >= 3:
                    break

            # fallback: if nothing selected, use first 200 chars of top hit
            if not selected:
                top_text = clean_text_for_display(hits[0].get("text", "")) if hits else ""
                snippet = top_text[:200].rstrip()
                answer = snippet + ("..." if len(top_text) > 200 else "")
            else:
                # join with separators; keep it short
                answer = " • ".join(selected)
                if len(answer) > 400:
                    answer = answer[:397].rstrip() + "..."

        # final response
        return Response({"answer": answer, "hits": hits})