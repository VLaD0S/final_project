from django import forms

from .models import Image


class PostForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = {
            "image",
        }

        def save(self, *args, **kwargs):
            instance = super(Image, self).save(commit=False)
            return instance

