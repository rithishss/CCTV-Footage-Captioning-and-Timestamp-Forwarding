import os
import json
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from feature_extraction import video_feature_extractor 

MODEL_PATH = "lstm_model.keras"
TOKENIZER_PATH = "tokenizer.json"
SCALER_PATH = "scaler.pkl"
SVD_PATH = "svd.pkl"
MAX_LENGTH = 11

def loading_and_preprocessing(X_Features):

    print("Loading trained LSTM model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("LSTM model loaded successfully.")

    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer_data = f.read()
    tokenizer = tokenizer_from_json(tokenizer_data)
    print("Tokenizer loaded successfully.")

    print("Loading scaler and SVD...")
    scaler = joblib.load(SCALER_PATH)
    svd = joblib.load(SVD_PATH)
    print("Scaler and SVD loaded successfully.")

    print("Loading raw video features...")
    X_raw = X_Features
    print(f"Loaded raw feature array with shape: {X_raw.shape}")

    print("Applying SVD and scaling transformations...")

    n_samples, timesteps, n_features = X_raw.shape
    X_flat = X_raw.reshape(-1, n_features)
    X_reduced = svd.transform(X_flat)
    X_reduced = X_reduced.reshape(n_samples, timesteps, -1)

    num_videos, num_frames, feat_dim = X_reduced.shape
    X_scaled_flat = X_reduced.reshape(-1, feat_dim)

    X_scaled_flat = scaler.transform(X_scaled_flat)
    X_reduced = X_scaled_flat.reshape(num_videos, num_frames, feat_dim)

    return X_reduced

def features_concatenator(spatial_features, temporal_features):
    X_features = []
    spatial_feat = spatial_features
    temporal_feat = temporal_features

    if len(temporal_feat.shape) == 1:
        temporal_feat_expanded = np.tile(temporal_feat, (spatial_feat.shape[0], 1))
    else:
        temporal_feat_expanded = temporal_feat

    hybrid_feat = np.concatenate([spatial_feat, temporal_feat_expanded], axis=-1)
    X_features.append(hybrid_feat)

    min_feature_dim = min([f.shape[1] for f in X_features])
    X_fixed = [f[:, :min_feature_dim] for f in X_features]

    max_time_steps = max([f.shape[0] for f in X_fixed])
    X_padded = []

    for f in X_fixed:
        if f.shape[0] < max_time_steps:
            pad_width = ((0, max_time_steps - f.shape[0]), (0, 0))
            f_padded = np.pad(f, pad_width, mode='constant', constant_values=0)
        else:
            f_padded = f
        X_padded.append(f_padded)

    X_features = np.array(X_padded)
    return X_features

def generate_caption_greedy(model, video_feature, tokenizer, max_length=11):
    start_token = tokenizer.word_index.get('start')
    end_token = tokenizer.word_index.get('end')
    inv_map = {v: k for k, v in tokenizer.word_index.items()}

    caption_seq = [start_token]
    for _ in range(max_length):
        decoder_input_seq = np.array(caption_seq)[None, :]
        preds = model.predict([video_feature[None, :, :], decoder_input_seq], verbose=0)
        next_token = np.argmax(preds[0, len(caption_seq) - 1, :])
        if next_token == end_token:
            break
        caption_seq.append(next_token)

    caption = [inv_map.get(i, "<unk>") for i in caption_seq if i not in [start_token, end_token]]
    return " ".join(caption)

def lstm_captioning(video):
    spatial_features, temporal_features = video_feature_extractor(video)
    X_Features = features_concatenator(spatial_features, temporal_features)
    X_reduced = loading_and_preprocessing(X_Features)

    generated_captions_greedy = {}

    for i in range(len(X_reduced)):
        caption = generate_caption_greedy(model, X_reduced[i], tokenizer, max_length=MAX_LENGTH)
        generated_captions_greedy[f"video_{i}"] = caption

    return generated_captions_greedy