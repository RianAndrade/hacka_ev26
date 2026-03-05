# sinan/serializers.py
from rest_framework import serializers


class SinanNotificationCSVImportSerializer(serializers.Serializer):
    file = serializers.FileField()
    batch_size = serializers.IntegerField(
        required=False,
        min_value=1,
        default=1000
    )
    dry_run = serializers.BooleanField(required=False, default=False)

    def validate_file(self, f):
        name = (getattr(f, "name", "") or "").lower()
        if not name.endswith(".csv"):
            raise serializers.ValidationError("Envie um arquivo .csv")
        return f
