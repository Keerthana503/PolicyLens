# backend/api/views.py
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

# -------------------------
# TF-IDF query view (lightweight)
# -------------------------
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

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
        mat_path = os.path.join(tfidf_dir, "matrix.npz")
        vocab_path = os.path.join(tfidf_dir, "vectorizer_vocab.json")
        meta_path = os.path.join(tfidf_dir, "metadata.json")
        if not (os.path.exists(mat_path) and os.path.exists(vocab_path) and os.path.exists(meta_path)):
            return Response({"detail":"TF-IDF index not available"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        metadata = json.load(open(meta_path, "r", encoding="utf-8"))
        vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
        X = sparse.load_npz(mat_path)

        # build vectorizer with saved vocabulary, then fit on the saved chunks to set idf_
        vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocab)
        # metadata is a list of dicts saved earlier; build list of texts
        texts = [m["text"] for m in metadata]
        # fit vectorizer to compute idf_ (fast because texts are small)
        vectorizer.fit(texts)
        # now transform the question safely
        q_vec = vectorizer.transform([question])  # shape (1, n_features)
        scores = (q_vec.dot(X.T)).toarray()[0]
        ranked = np.argsort(-scores)
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            item = metadata[idx]
            if item.get("doc_id") == doc.id:
                hits.append({"score": float(scores[idx]), "page": item["page"], "text": item["text"]})
            if len(hits) >= top_k:
                break

        # simple short summary from top hits (first sentence of each)
        summary_lines = []
        for h in hits[:3]:
            snippet = h["text"]
            sentence = snippet.split(".")[0].strip()
            if sentence:
                summary_lines.append(sentence)
        summary = " • ".join(summary_lines) if summary_lines else ""

        return Response({"answer": summary, "hits": hits})