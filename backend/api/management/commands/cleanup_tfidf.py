from django.core.management.base import BaseCommand
import os, glob
from django.conf import settings

class Command(BaseCommand):
    help = "Remove TF-IDF artifacts for a document or all docs"

    def add_arguments(self, parser):
        parser.add_argument("--doc-id", type=int, help="Document id to clean (optional)")
        parser.add_argument("--all", action="store_true", help="Remove all tfidf_index files (use carefully)")

    def handle(self, *args, **options):
        tfidf_dir = os.path.join(settings.BASE_DIR, "tfidf_index")
        if options.get("all"):
            pattern = os.path.join(tfidf_dir, "doc_*.*")
        elif options.get("doc_id"):
            base = f"doc_{options['doc_id']}"
            pattern = os.path.join(tfidf_dir, f"{base}_*")
        else:
            self.stderr.write(self.style.ERROR("Provide --doc-id or --all"))
            return

        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
                self.stdout.write(self.style.SUCCESS(f"Removed {f}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Failed to remove {f}: {e}"))