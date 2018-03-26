import load_model

def get_prediction(image):
    image = "media/"+image

    prediction, label = load_model.print_prediction('inception_1', image, 299, True)

    return prediction, label


