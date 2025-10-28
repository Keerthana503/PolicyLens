# backend/api/management/commands/ingest_document_hf.py
import os, json, numpy as np, faiss
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
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
    help = "Ingest a Document by ID using sentence-transformers & FAISS (no OpenAI)."

    def add_arguments(self, parser):
        parser.add_argument("document_id", type=int)

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
            # extract text
            reader = PdfReader(doc.file.path)
            all_chunks = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if not text.strip():
                    continue
                page_chunks = chunk_text(text)
                for c in page_chunks:
                    all_chunks.append({"page": i + 1, "text": c})

            if not all_chunks:
                self.stdout.write(self.style.WARNING("No text extracted from PDF"))
                doc.status = "failed"
                doc.save()
                return

            # create embeddings
            model = SentenceTransformer(EMBED_MODEL_NAME)
            texts = [c["text"] for c in all_chunks]
            self.stdout.write(f"Creating embeddings for {len(texts)} chunks...")
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            embeddings = embeddings.astype("float32")
            dim = embeddings.shape[1]
            self.stdout.write(f"Embedding dimension: {dim}")

            # prepare faiss dir
            faiss_dir = os.path.join(settings.BASE_DIR, "faiss_index")
            os.makedirs(faiss_dir, exist_ok=True)
            index_path = os.path.join(faiss_dir, "index.faiss")
            meta_path = os.path.join(faiss_dir, "metadata.json")

            # load or create index
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if index.d != dim:
                    self.stdout.write(self.style.WARNING("Dimension mismatch: recreating index"))
                    index = faiss.IndexFlatL2(dim)
                    metadata = []
            else:
                index = faiss.IndexFlatL2(dim)
                metadata = []

            # add vectors & metadata
            index.add(embeddings)
            for c in all_chunks:
                metadata.append({
                    "doc_id": doc.id,
                    "page": c["page"],
                    "text": c["text"][:2000]
                })

            faiss.write_index(index, index_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            doc.status = "ready"
            doc.save()
            self.stdout.write(self.style.SUCCESS(f"Document {doc.id} ingested. Index saved to {index_path}"))
        except Exception as e:
            doc.status = "failed"
            doc.save()
            self.stderr.write(self.style.ERROR(f"Ingestion failed: {e}"))
            raise