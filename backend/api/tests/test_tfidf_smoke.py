from django.test import TestCase, Client
from django.contrib.auth.models import User
from api.models import Document
import os
from django.conf import settings

class TfIdfSmokeTests(TestCase):
    def setUp(self):
        self.u = User.objects.create_user("tuser","t@x.com","pass")
        self.client = Client()
        self.client.force_login(self.u)

    def test_validate_artifacts_missing(self):
        doc = Document.objects.create(
            owner=self.u,
            file="media/documents/dummy.pdf",
            status="uploaded"
        )

        from api.utils.tfidf_utils import validate_artifacts
        matrix = os.path.join(settings.BASE_DIR, "tfidf_index", "doc_999_matrix.npz")
        ok, reason, details = validate_artifacts(matrix)

        self.assertFalse(ok)
        self.assertIn("matrix_missing", reason)