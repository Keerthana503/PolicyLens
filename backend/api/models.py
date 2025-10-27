from django.db import models
from django.conf import settings

class Document(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="documents/%Y/%m/%d/")
    original_filename = models.CharField(max_length=512, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=32, default="uploaded")  # uploaded, processing, ready, failed
    pages = models.IntegerField(null=True, blank=True)  # optional, fill later if you extract pages

    def _str_(self):
        return f"{self.title or self.original_filename} ({self.owner.username})"