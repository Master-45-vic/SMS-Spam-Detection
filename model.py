import pickle
import numpy as np

def load_classifier(path):
    return pickle.load(open(path, "rb"))

def predict_message(model, vector_input):
    vector_input = vector_input.reshape(1, -1).astype(np.float32)
    
    prediction = model.predict(vector_input)
    probability = model.predict_proba(vector_input)
    
    return prediction[0], probability[0]