import os
from io import StringIO
from sentence_transformers import SentenceTransformer, util
import torch

from feature_extraction import video_feature_extractor
from lstm_captioning import lstm_captioning

def convert_timestamp_to_seconds(timestamp_str):
    """Converts a 'HH:MM:SS.fff' timestamp string to total seconds."""
    parts = timestamp_str.split(':')
    seconds_parts = parts[2].split('.')
    h = int(parts[0])
    m = int(parts[1])
    s = int(seconds_parts[0])
    return h * 3600 + m * 60 + s

def search_video_captions_semantically(video_captions, query, similarity_threshold=0.4):
    """
    Downloads captions, performs a semantic search, and returns clickable timestamps.

    Args:
        video_captions (str): The captions generated.
        query (str): The search phrase or question.
        similarity_threshold (float): How similar the text needs to be (0.0 to 1.0).
                                      A good starting point is 0.5-0.6.
    Returns:
        list: A list of tuples, each containing (timestamp_link, caption_text).
    """
    caption_content = video_captions
    model = SentenceTransformer('all-MiniLM-L6-v2')

    caption_chunks = []
    chunk_size = 3
    for i in range(0, len(video_captions), chunk_size):
        chunk = video_captions[i:i+chunk_size]
        text = " ".join(c.text.replace('\n', ' ').strip() for c in chunk)
        start_time = chunk[0].start
        caption_chunks.append({'text': text, 'start': start_time})

    if not caption_chunks:
        print("No caption found!.")
        return []

    corpus_texts = [chunk['text'] for chunk in caption_chunks]
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    found_timestamps = []

    for i, score in enumerate(cosine_scores):
        if score > similarity_threshold:
            caption_info = caption_chunks[i]
            timestamp_str = caption_info['start']
            total_seconds = convert_timestamp_to_seconds(timestamp_str)

            found_timestamps.append(caption_info['text'])

    return found_timestamps

def streamlit_timestamping(video, search_word):
    captions = lstm_captioning(video)
    return search_video_captions_semantically(captions, search_word)