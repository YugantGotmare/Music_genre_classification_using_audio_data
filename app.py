import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam

# Load the saved model with custom optimizer
model = load_model('audio_classification_16_45_02 .hdf5', compile=False)

# Function to extract features from audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    except Exception as e:
        print(f"Error: {e}")
        return None
    return mfccs_scaled_features

# Streamlit UI
st.title("Music Genre Classification App")

uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        # Extract features from the uploaded audio file
        features = extract_features(uploaded_file)

        if features is not None:
            # Reshape features for model prediction
            features = np.expand_dims(features, axis=0)

            # Make prediction using the loaded model
            prediction = model.predict_classes(features)
            genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                             5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

            # Display the predicted genre
            st.success(f"Predicted Genre: {genre_mapping[prediction[0]]}")
        else:
            st.error("Error extracting features from the audio file.")

    except Exception as e:
        st.error(f"Error: {e}")
