from django.db import models
from __future__ import unicode_literals
from datetime import datetime
class Website(models.Model):
    title = models.CharField(max_length=200)
    text=models.TextField();
    created_at=models.DateTimeField(default=datetime.now   )