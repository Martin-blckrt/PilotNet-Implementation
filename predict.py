import tensorflow as tf
import keras


def predict(self, data, given_model='default'):
    if given_model != 'default':
        try:
            # load the model
            model = keras.models.load_model(f'models/{given_model}', custom_objects={"tf": tf})
        except:
            raise Exception('An unexpected error occured when loading the saved model. Please rerun...')
    else:
        model = self.model
    # predict using the model
    predictions = model.predict(data.image)
    return predictions
