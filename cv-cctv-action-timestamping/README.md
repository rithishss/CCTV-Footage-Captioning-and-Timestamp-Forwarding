# CCTV Footage Captioning and Timestamp Forwarding

A CCTV footage captioning product aimed at malicious-intent action tasks, with a timestamp forwarding feature based on Semantic Search.

## Dataset

Dataset Link : [CCTV Action Recognition Dataset](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset)

The CCTV Action Recognition Dataset is a collection of short video clips designed for training models to recognize human actions in surveillance footage. It consists of 13 action categories, including Fall, Gun, Hit, and Struggle, with 200 clips per category. The dataset also includes pre-generated training and testing splits and uses a structured naming convention that indicates the clip's source, original video, and action type, making it a practical resource for developing and benchmarking action recognition models. This dataset videos are named after the convention "{Name_of_source}{Name_of_video_type}{Action_Category}". Each video is nearly about 3 to 4 seconds long, just indicating the events. The videos weren't formally captioned.

Since the captions were not present initially, an attempt was made to caption the videos according to the scene in detail. This would help the model learn better for more details in the video. It would also help understand the specifics in the actions. For example, adding information about background details (time of the day, proximity to objects), number of people involved (the main action and the surrounding), The specifics of the action (Eg: stabbing can be done in different places. More information about the instrument used, where it affected the other person and their response to it), etc.

## Methodology

![Architecture Diagram](./Images/CV-Arch.png)

### I. Feature Extraction and Image Enhancement

The dataset was first balanced by trimming videos per category, after which each frame was enhanced with ZeroDCE (Deep Curve Estimation) to improve local contrast for robust feature detection. Since simple CLAHE based image modification would struggle with varying subjects, color of clothing, etc., we resorted to a Deep Learning based approach for image enhancement. ZeroDCE (Deep Curve Estimation) takes into account multiple loss curves such as Color constancy loss, spatial consistency loss, etc. This has been adopted and integrated into the complete workflow, thus enabling better feature extraction. 

This was followed by spatial and temporal feature extraction. A hybrid approach captured both scene content and motion: a VGG16 CNN extracted spatial features, while a Fast Fourier Transform (FFT) on frame differences extracted temporal motion patterns. Features were fused before it was fed into the captioning module.

### II. Bi-LSTM + Attention Layer for Captioning

The original video features had 25,117 dimensions per timestep, which made the model computationally expensive and increased the risk of overfitting. Singular Value Decomposition (SVD) was used to reduce the features into 1500 components capturing >99.999% total variance. The features were normalized using StandardScaler, to make sure each dimension contributes equally and keep the training numerically stable.

The encoder uses a Bidirectional LSTM with dropout to capture temporal patterns from reduced video features. The decoder embeds caption tokens, processes them with an LSTM, and applies an attention mechanism to focus on relevant video frames. The final output layer predicts the next word in the caption, achieving ~98% training accuracy and ~61% validation accuracy.

### III. Timestamp Forwarding

This was done using Semantic search on the generated captions and the query. The steps are as follows:

* Done for extracting the occurrences of a phrase or a word based on its meaning. This helps find similar events that could help track down the occurrence of all possible activities.
* The caption generated is converted to a list of caption objects, which are then grouped into three-caption-line text chunks to provide broader context for semantic analysis.
* A pre-trained SentenceTransformer model (all-MiniLM-L6-v2) is loaded, which is capable of converting text into high-dimensional numerical vector (an embedding).
* The cosine similarity between the query embedding and every caption_chunk embedding is calculated, resulting in a score for each chunk indicating its semantic closeness to the query. 
* This is compared to a threshold. If it is above the threshold, it is a considered a match. For each relevant match, its start time is returned. 
