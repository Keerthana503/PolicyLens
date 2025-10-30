# backend/api/management/commands/ingest_document_tfidf_safe.py
import os, json, re
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from pdfminer.high_level import extract_text

# Conservative settings to protect memory
CHUNK_SIZE = 600          # smaller chunks
CHUNK_OVERLAP = 120
MAX_PAGE_CHARS = 200_000  # hard cap per page
MAX_FEATURES = 5000       # smaller vocab to reduce RAM

def clean_text(s: str) -> str:
    if not s:
        return ""
    # remove NULs and non-printables
    s = s.replace("\x00", " ")
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

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
        # move with overlap
        start = end - overlap
        if start <= 0:
            start = 0
        if start >= L:
            break
    return chunks

class Command(BaseCommand):
    help = "Ingest a Document by ID using TF-IDF with pdfminer and memory-safe caps."

    def add_arguments(self, parser):
        parser.add_argument("document_id", type=int)

    def handle(self, *args, **options):
        doc_id = options["document_id"]
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Document {doc_id} not found"))
            return

        self.stdout.write(f"Ingesting (TF-IDF SAFE) Document id={doc.id} file={doc.file.path}")
        doc.status = "processing"
        doc.save()

        try:
            # pdfminer: extract page-by-page to avoid huge strings
            # We don't have to open the file manually; pdfminer reads paths directly.
            # Count pages by trying until failure would be heavy, so we just extract full text and split on form-feed as fallback.
            # Better: call extract_text once and split on \f, which pdfminer inserts between pages.
            raw = extract_text(doc.file.path) or ""
            if not raw.strip():
                self.stderr.write(self.style.ERROR("No extractable text found (likely a scanned image PDF)."))
                doc.status = "failed"
                doc.save()
                return

            pages = raw.split("\f")  # pdfminer uses form-feed between pages
            all_texts = []
            metadata = []

            for i, page_text in enumerate(pages, start=1):
                t = clean_text(page_text)
                if not t:
                    continue

                # hard cap per page to guard against broken PDFs
                if len(t) > MAX_PAGE_CHARS:
                    t = t[:MAX_PAGE_CHARS]

                chunks = chunk_text(t)
                for c in chunks:
                    all_texts.append(c)
                    metadata.append({"doc_id": doc.id, "page": i, "text": c})

            if not all_texts:
                self.stderr.write(self.style.ERROR("No usable text after cleaning/caps."))
                doc.status = "failed"
                doc.save()
                return

            # Build TF-IDF (sparse, low memory)
            vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
            X = vectorizer.fit_transform(all_texts)

            tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
            os.makedirs(tfidf_dir, exist_ok=True)

            # save vocab / metadata / sparse matrix
            with open(os.path.join(tfidf_dir, "vectorizer_vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vectorizer.vocabulary_, f)

            with open(os.path.join(tfidf_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            sparse.save_npz(os.path.join(tfidf_dir, "matrix.npz"), X)

            doc.status = "ready"
            doc.save()
            self.stdout.write(self.style.SUCCESS(f"Document {doc.id} ingested with TF-IDF SAFE. Chunks: {len(all_texts)}"))
            self.stdout.write(self.style.SUCCESS(f"Saved index to {tfidf_dir}"))
        except Exception as e:
            doc.status = "failed"
            doc.save()
            self.stderr.write(self.style.ERROR(f"Ingestion failed: {e}"))
            raise