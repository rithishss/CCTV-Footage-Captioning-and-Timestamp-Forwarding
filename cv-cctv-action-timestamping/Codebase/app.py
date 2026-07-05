import streamlit as st
import random
import base64

from feature_extraction import video_feature_extractor
from lstm_captioning import lstm_captioning
from timestamping import streamlit_timestamping

st.set_page_config(page_title="CCTV Footage Search", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
    }
    .main {
        background: transparent;
    }
    h1 {
        text-align: center;
        font-size: 3.5rem !important;
        color: #00FFFF;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px #00ffff80;
    }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #00FFFF;
        background-color: #101820;
        color: white;
    }
    .stButton button {
        border-radius: 12px;
        background-color: #00FFFF;
        color: black;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #00cccc;
        transform: scale(1.05);
    }
    .timestamp-btn {
        margin: 6px;
        padding: 10px 16px;
        border: none;
        border-radius: 10px;
        background: #00FFFF;
        color: black;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s;
    }
    .timestamp-btn:hover {
        background: #00cccc;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Advanced CCTV Video Captioning</h1>", unsafe_allow_html=True)
st.write("#### Powered by AI to detect and find the exact moment of the actions in real-time surveillance footage")

col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("Drop the footage to pass it for captioning", type=["mp4"])

with col2:
    search_phrase = st.text_input("Find specific actions in your footage", placeholder="e.g. fight, explosion, idle...")
    run_button = st.button("Search")

def get_timestamps_from_model(video, query):
    return streamlit_timestamping(video, query)

if video_file:
    video_bytes = video_file.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    video_id = "videoPlayer"

    timestamps = []
    if run_button:
        video_file.seek(0)
        timestamps = get_timestamps_from_model(video_file, search_phrase)

    buttons_html = ""
    if timestamps:
        for ts in timestamps:
            buttons_html += f'<button class="timestamp-btn" onclick="seekTo({ts})">{ts}s</button>'
    else:
        if run_button:
            buttons_html = "<div style='color:#ffd2b3;font-style:italic;'>No relevant moments found.</div>"

    full_html = f"""
    <div style='text-align:center'>
        <video id="{video_id}" width="720" controls preload="metadata" style="border-radius:12px; margin-top:15px;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div style="margin-top:12px;">{buttons_html}</div>
    </div>

    <script>
    function seekTo(t) {{
        const v = document.getElementById('{video_id}');
        if (!v) return;
        // Wait for metadata before seeking
        if (v.readyState > 0 && !isNaN(v.duration)) {{
            const safeT = Math.min(t, Math.max(0, v.duration - 0.1));
            v.currentTime = safeT;
            v.play();
        }} else {{
            v.addEventListener('loadedmetadata', () => {{
                const safeT = Math.min(t, Math.max(0, v.duration - 0.1));
                v.currentTime = safeT;
                v.play();
            }}, {{ once: true }});
        }}
    }}
    </script>
    """
    st.markdown(video_html, unsafe_allow_html=True)

    if run_button:
        st.divider()
        st.subheader(f"Results for: **{search_phrase}**")
        video_file.seek(0)
        timestamps = get_timestamps_from_model(video_file, search_phrase)

        if timestamps:
            st.success(f"Found {len(timestamps)} relevant instances!")

            js_buttons = ""
            for ts in timestamps:
                js_buttons += f"""
                <button class="timestamp-btn" onclick="document.getElementById('{video_id}').currentTime={ts};
                document.getElementById('{video_id}').play();">{ts}s</button>
                """

            st.markdown(f"<div style='text-align:center'>{js_buttons}</div>", unsafe_allow_html=True)
            st.info("Timestamps lead you to the exact occurence of the action in the video!.")
        else:
            st.warning("No relevant moments found.")
