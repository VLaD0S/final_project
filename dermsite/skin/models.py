from django.db import models

# Create your models here.


# diagnosis class
class Dig(models.Model):

    iden_key = models.CharField(max_length=256)
    image_path = models.CharField(max_length=256)
    top_prediction = models.FloatField()

    def __str__(self):
        return self.image_path
