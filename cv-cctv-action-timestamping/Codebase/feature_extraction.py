import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import os
import glob
from tqdm import tqdm
import sys

from image_enhancement import ZeroDCE

def apply_zerodce(frame):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a frame.
    Enhances the contrast of the grayscale image.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_frame

def extract_spatial_features(frame, model):
    """
    Extract spatial features from a frame using a pre-trained CNN model.
    """
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=0)
    return features.flatten()

def analyze_temporal_patterns(frames):
    """
    Analyze temporal motion patterns using Fast Fourier Transform (FFT).
    """
    if len(frames) < 2:
        return np.array([])
    frame_diffs = [cv2.absdiff(frames[i+1], frames[i]) for i in range(len(frames)-1)]
    gray_diffs = [cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) for diff in frame_diffs]
    if not gray_diffs:
        return np.array([])
    diff_stack = np.stack(gray_diffs, axis=0)
    fft_result = np.fft.fftn(diff_stack)
    fft_magnitude = np.abs(np.fft.fftshift(fft_result))
    return np.mean(fft_magnitude, axis=(1,2))

def process_video(video, model):
    """
    Function to process a single video and return the features.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Error: Could not open video")
        return

    frame_count = 0
    processed_frames_for_fft = []
    all_spatial_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        enhanced_frame = apply_zerodce(frame)
        spatial_features = extract_spatial_features(enhanced_frame, model)
        all_spatial_features.append(spatial_features)
        processed_frames_for_fft.append(enhanced_frame)
        frame_count += 1

    cap.release()
    print(f"Finished processing {frame_count} frames.")

    temporal_features = analyze_temporal_patterns(processed_frames_for_fft)
    return [all_spatial_features, temporal_features]

def video_feature_extractor(video):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    return process_video(video, model)