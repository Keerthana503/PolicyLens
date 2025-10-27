from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework import serializers
from  .models import Document

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, min_length=6)

    class Meta:
        model = User
        fields = ("id", "username", "email", "password")
        extra_kwargs = {"email": {"required": True}}

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"]
        )
        return user

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username", "email")

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ("id", "owner", "title", "original_filename", "file", "uploaded_at", "status")
        read_only_fields = ("id", "owner", "uploaded_at", "status", "original_filename")

    def create(self, validated_data):
        # owner will be set in view; also capture original filename
        request = self.context.get("request")
        uploaded_file = request.FILES.get("file") if request else None
        if uploaded_file:
            validated_data["original_filename"] = uploaded_file.name
        doc = super().create(validated_data)
        return doc