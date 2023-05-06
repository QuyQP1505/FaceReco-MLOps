import numpy as np


def predict(model, feautures):
    new_features = np.array(feautures)
    prediction = model.predict([new_features])[0]

    return prediction

def recognize_faces(images, model, feautures):
    predictions = []
    for image in images:
        feautures = feautures.load(image_path=image)
        prediction = predict(model, feautures)
        predictions.append(prediction)

    return predictions