from django.urls import path
from .views import RegisterView, MeView, DocumentListCreateView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    # Auth
    path("auth/register/", RegisterView.as_view(), name="auth_register"),
    path("auth/login/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # Get current user
    path("auth/me/", MeView.as_view(), name="auth_me"),
     # Documents
    path("documents/", DocumentListCreateView.as_view(), name="documents_list_create"),
]