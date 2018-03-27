import load_model
from . models import Image


def get_prediction(image, id):

    if image:
        path = "media/"+image
        prediction, label = load_model.print_prediction('inception_1', path, 299, True)

        print(id)
        data = Image.objects.get(pk=id)
        data.label = label
        print(data.label)
        data.top_prediction = prediction
        data.save()

    return data


