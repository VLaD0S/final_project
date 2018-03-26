from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse
from .forms import PostForm
from .prediction import get_prediction as predict


# index page
def index(request):
    form = PostForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        instance = form.save()
        print(instance.id)
        print(instance.image)
        #instance = predict(instance.image)
        print(instance.label)
        print(instance.id)
        instance.save()
        messages.success(request, "Successfully Created")
        return HttpResponse(index)

    context = {"form": form}
    return render(request, 'post_form.html', context)

