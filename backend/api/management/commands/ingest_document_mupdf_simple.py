import os, json, re
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Document
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

MAX_PAGE_CHARS = 100_000
MAX_FEATURES = 3000

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

class Command(BaseCommand):
    help = "Ingest using PyMuPDF but treat each page as a single chunk (memory-safe)."

    def add_arguments(self, parser):
        parser.add_argument("document_id", type=int)

    def handle(self, *args, **options):
        doc_id = options["document_id"]
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Document {doc_id} not found"))
            return

        self.stdout.write(f"Ingesting (MuPDF simple) Document id={doc.id} file={doc.file.path}")
        doc.status = "processing"
        doc.save()

        try:
            docfile = fitz.open(doc.file.path)
            all_texts = []
            metadata = []
            for i in range(docfile.page_count):
                page = docfile.load_page(i)
                text = page.get_text("text") or ""
                text = clean_text(text)
                if not text:
                    continue
                if len(text) > MAX_PAGE_CHARS:
                    text = text[:MAX_PAGE_CHARS]
                all_texts.append(text)
                metadata.append({"doc_id": doc.id, "page": i+1, "text": text})
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
                vocab_clean = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
                json.dump(vocab_clean, f)
            with open(os.path.join(tfidf_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            sparse.save_npz(os.path.join(tfidf_dir, "matrix.npz"), X)

            doc.status = "ready"
            doc.save()
            self.stdout.write(self.style.SUCCESS(f"Document {doc.id} ingested (MuPDF simple). Chunks (pages): {len(all_texts)}"))
            self.stdout.write(self.style.SUCCESS(f"Saved index to {tfidf_dir}"))
        except Exception as e:
            doc.status = "failed"
            doc.save()
            self.stderr.write(self.style.ERROR(f"Ingestion failed: {e}"))
            raise
