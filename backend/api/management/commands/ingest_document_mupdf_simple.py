# ingest_document_mupdf_simple.py
import os
import re
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Document

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# Tunable chunking params
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MAX_PAGE_CHARS = 200_000
MAX_FEATURES = 30000  # limit TF-IDF vocab size to avoid huge memory on large corpora


def clean_text(s: str) -> str:
    if not s:
        return ""
    # remove nulls and PDF (cid:123) artifacts, keep most unicode printable
    s = s.replace("\x00", " ")
    s = re.sub(r"\(cid:\d+\)", " ", s)
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text or ""
    n = len(text)
    if n == 0:
        return []
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        # move start forward with overlap
        start = max(0, end - overlap)
    return chunks


class Command(BaseCommand):
    help = "Ingest a document (MuPDF) and build per-document TF-IDF index files."

    def add_arguments(self, parser):
        parser.add_argument("document_id", type=int, help="ID of Document to ingest")

    def handle(self, *args, **options):
        doc_id = options["document_id"]
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Document {doc_id} not found"))
            return

        self.stdout.write(f"Ingesting Document id={doc.id} file={doc.file.path}")
        doc.status = "processing"
        doc.save()

        try:
            pdf = fitz.open(doc.file.path)
            all_chunks = []
            metadata = []  # list of dicts: {doc_id, page, chunk_index, text}
            for p in range(pdf.page_count):
                page = pdf.load_page(p)
                text = page.get_text("text") or ""
                text = clean_text(text)
                if not text:
                    continue
                # guard against huge single-page text (rare)
                if len(text) > MAX_PAGE_CHARS:
                    text = text[:MAX_PAGE_CHARS]
                chunks = chunk_text(text)
                for i, c in enumerate(chunks):
                    all_chunks.append(c)
                    metadata.append({
                        "doc_id": doc.id,
                        "page": p + 1,
                        "chunk_index": i,
                        "text": c
                    })
            pdf.close()

            if not all_chunks:
                self.stderr.write(self.style.ERROR("No text extracted from document — nothing to index."))
                doc.status = "failed"
                doc.save()
                return

            # Fit TF-IDF on chunks (fast for small documents)
            vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
            X = vectorizer.fit_transform(all_chunks)  # sparse matrix

            # Save per-document files
            tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
            os.makedirs(tfidf_dir, exist_ok=True)

            base = f"doc_{doc.id}"
            matrix_path = os.path.join(tfidf_dir, f"{base}_matrix.npz")
            vocab_path = os.path.join(tfidf_dir, f"{base}_vocab.json")
            meta_path = os.path.join(tfidf_dir, f"{base}_metadata.json")

            sparse.save_npz(matrix_path, X)
            # convert numpy ints to native Python ints so JSON can serialize
            vocab_serializable = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
            with open(vocab_path, "w", encoding="utf-8") as vf:
                json.dump(vocab_serializable, vf, ensure_ascii=False, indent=2)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, ensure_ascii=False, indent=2)

            doc.status = "ready"
            doc.save()

            self.stdout.write(self.style.SUCCESS(f"Document {doc.id} ingested: chunks={len(all_chunks)}"))
            self.stdout.write(self.style.SUCCESS(f"Saved: {matrix_path}"))
            self.stdout.write(self.style.SUCCESS(f"Saved: {vocab_path}"))
            self.stdout.write(self.style.SUCCESS(f"Saved: {meta_path}"))

        except Exception as e:
            doc.status = "failed"
            doc.save()
            self.stderr.write(self.style.ERROR(f"Ingestion failed: {e}"))
            raise