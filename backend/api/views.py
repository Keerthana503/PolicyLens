from rest_framework import generics, permissions
from .serializers import RegisterSerializer, UserSerializer, DocumentSerializer
from django.contrib.auth.models import User
from .models import Document

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer
    pass

class MeView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    pass

    def get_object(self):
        return self.request.user

class DocumentListCreateView(generics.ListCreateAPIView):
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # users only see their own docs
        return Document.objects.filter(owner=self.request.user).order_by("-uploaded_at")

    def perform_create(self, serializer):
        # set owner automatically and save file info
        serializer.save(owner=self.request.user, status="uploaded")