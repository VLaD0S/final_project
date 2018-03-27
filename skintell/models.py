from django.db import models

# Create your models here.


# diagnosis class
class Image(models.Model):
    image = models.FileField(null=True)
    top_prediction = models.FloatField(null=True)
    label = models.CharField(max_length=256, null=True)

    def save(self, *args, **kwargs):
        instance = super(Image, self).save(*args, **kwargs)
        return instance



