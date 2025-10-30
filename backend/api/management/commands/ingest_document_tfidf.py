# backend/api/management/commands/ingest_document_tfidf.py
import os, json
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

class Command(BaseCommand):
    help = "Ingest a Document by ID using TF-IDF (low memory)."

    def add_arguments(self, parser):
        parser.add_argument("document_id", type=int)

    def handle(self, *args, **options):
        doc_id = options["document_id"]
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Document {doc_id} not found"))
            return

        self.stdout.write(f"Ingesting (TF-IDF) Document id={doc.id} file={doc.file.path}")
        doc.status = "processing"
        doc.save()

        try:
            reader = PdfReader(doc.file.path)
            all_texts = []
            metadata = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                page_chunks = chunk_text(text)
                for c in page_chunks:
                    metadata.append({"doc_id": doc.id, "page": i+1, "text": c})
                    all_texts.append(c)

            if not all_texts:
                self.stderr.write(self.style.ERROR("No extractable text found"))
                doc.status = "failed"
                doc.save()
                return

            # Compute TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
            X = vectorizer.fit_transform(all_texts)  # sparse matrix
            # Save vectorizer vocabulary and metadata
            tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
            os.makedirs(tfidf_dir, exist_ok=True)
            vec_path = os.path.join(tfidf_dir, "vectorizer_vocab.json")
            with open(vec_path, "w", encoding="utf-8") as f:
                json.dump(vectorizer.vocabulary_, f)
            meta_path = os.path.join(tfidf_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            # save sparse matrix as npz
            from scipy import sparse
            mat_path = os.path.join(tfidf_dir, "matrix.npz")
            sparse.save_npz(mat_path, X)

            doc.status = "ready"
            doc.save()
            self.stdout.write(self.style.SUCCESS(f"Document {doc.id} ingested with TF-IDF. Chunks: {len(all_texts)}"))
            self.stdout.write(self.style.SUCCESS(f"Saved index to {tfidf_dir}"))
        except Exception as e:
            doc.status = "failed"
            doc.save()
            self.stderr.write(self.style.ERROR(f"Ingestion failed: {e}"))
            raise