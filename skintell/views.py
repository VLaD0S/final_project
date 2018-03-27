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

        data = predict(str(instance.image), int(instance.id))
        label = data.label

        prediction = data.top_prediction
        context = {"data_label": label,
                   "data_pred" : prediction,
                   "form": form}
        
        return render(request, "post_form.html", context)

    context = {"form": form}
    return render(request, 'post_form.html', context)

