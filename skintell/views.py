from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse
from .forms import PostForm
from .prediction import get_prediction as predict


# index page
def index(request):
    form = PostForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        instance = form.save(commit=False)
        instance.save()

        predict(str(instance.image), int(instance.id))
        context = {id: "id"}
        messages.success(request, "Successfully Created")
        return render(index)

    context = {"form": form}
    return render(request, 'post_form.html', context)

