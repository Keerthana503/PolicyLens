# backend/api/management/commands/ingest_document_mupdf.py
import os, json
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Document
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MAX_PAGE_CHARS = 100_000
MAX_FEATURES = 5000

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
        if start <= 0:
            start = 0
        if start >= L:
            break
    return chunks

class Command(BaseCommand):
    help = "Ingest using PyMuPDF (fitz) to extract page text safely and build TF-IDF index."

    def add_arguments(self, parser):
        parser.add_argument("document_id", type=int)

    def handle(self, *args, **options):
        doc_id = options["document_id"]
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Document {doc_id} not found"))
            return

        self.stdout.write(f"Ingesting (MuPDF) Document id={doc.id} file={doc.file.path}")
        doc.status = "processing"
        doc.save()

        try:
            docfile = fitz.open(doc.file.path)
            all_texts = []
            metadata = []
            for i in range(docfile.page_count):
                page = docfile.load_page(i)
                text = page.get_text("text") or ""
                text = text.strip()
                if not text:
                    continue
                if len(text) > MAX_PAGE_CHARS:
                    text = text[:MAX_PAGE_CHARS]
                chunks = chunk_text(text)
                for c in chunks:
                    metadata.append({"doc_id": doc.id, "page": i+1, "text": c})
                    all_texts.append(c)
            docfile.close()

            if not all_texts:
                self.stderr.write(self.style.ERROR("No text extracted (page texts empty)"))
                doc.status = "failed"
                doc.save()
                return

            vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
            X = vectorizer.fit_transform(all_texts)

            tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
            os.makedirs(tfidf_dir, exist_ok=True)
            with open(os.path.join(tfidf_dir, "vectorizer_vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vectorizer.vocabulary_, f)
            with open(os.path.join(tfidf_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            sparse.save_npz(os.path.join(tfidf_dir, "matrix.npz"), X)

            doc.status = "ready"
            doc.save()
            self.stdout.write(self.style.SUCCESS(f"Document {doc.id} ingested (MuPDF). Chunks: {len(all_texts)}"))
        except Exception as e:
            doc.status = "failed"
            doc.save()
            self.stderr.write(self.style.ERROR(f"Ingestion failed: {e}"))
            raise