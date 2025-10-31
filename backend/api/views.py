# backend/api/views.py
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import RegisterSerializer, UserSerializer, DocumentSerializer
from .models import Document


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
        
class QueryTfIdfAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        data = request.data
        doc_id = data.get("doc_id")
        question = data.get("question")
        top_k = int(data.get("top_k", 5))

        if not doc_id or not question:
            return Response({"detail":"doc_id and question are required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({"detail":"Document not found"}, status=status.HTTP_404_NOT_FOUND)

        if doc.owner_id != request.user.id:
            return Response({"detail":"Forbidden"}, status=status.HTTP_403_FORBIDDEN)

        tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
        base = f"doc_{doc.id}"
        matrix_path = os.path.join(tfidf_dir, base + "_matrix.npz")
        vect_path = os.path.join(tfidf_dir, base + "_vectorizer.pkl")
        meta_path = os.path.join(tfidf_dir, base + "_metadata.json")

        if not (os.path.exists(matrix_path) and os.path.exists(vect_path) and os.path.exists(meta_path)):
            return Response({"detail":"TF-IDF index for this document is not available. Run ingestion."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # load artifacts
        try:
            vectorizer = pickle.load(open(vect_path, "rb"))
            X = sparse.load_npz(matrix_path)
            metadata = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception as e:
            return Response({"detail": f"Failed to load index: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # transform question and score
        q_vec = vectorizer.transform([question])
        scores = (q_vec.dot(X.T)).toarray()[0]  # shape (n_chunks,)
        ranked = np.argsort(-scores)
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            item = metadata[idx]
            if item.get("doc_id") == doc.id:
                hits.append({
                    "score": float(scores[idx]),
                    "page": item.get("page"),
                    "chunk_index": item.get("chunk_index"),
                    "text": item.get("text")
                })
            if len(hits) >= top_k:
                break

        # ------------------------
        # Build a short, clean answer (generic)
        # ------------------------
        import re
        def clean_text_for_display(s: str) -> str:
            if not s:
                return ""
            s = re.sub(r"\(cid:\d+\)", " ", s)
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        if not hits:
            # no TF-IDF hits: return empty answer (caller can try rephrasing)
            return Response({"answer": "", "hits": []})

        # prefer sentences that include question keywords
        best_text = clean_text_for_display(hits[0]["text"])
        sentences = re.split(r'(?<=[\.\?\!])\s+', best_text)
        q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]

        matched = []
        if q_words:
            for s in sentences:
                sl = s.lower()
                if any(q in sl for q in q_words):
                    matched.append(s.strip())
                if len(matched) >= 3:
                    break

        if not matched:
            matched = [s.strip() for s in sentences[:3] if s.strip()]

        summary = " • ".join(matched) if matched else best_text[:400]
        if len(summary) > 400:
            summary = summary[:397].rstrip() + "..."

        return Response({"answer": summary, "hits": hits})