from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from cvscout.video import VideoReader


st.set_page_config(page_title="Scoutpost CV Scout", layout="wide")
st.title("FRC CV Scouting (v1 scaffold)")

video_path = st.text_input("Local video path", value="")
load = st.button("Load")

if "reader" not in st.session_state:
    st.session_state.reader = None
    st.session_state.video_path = None

if load:
    try:
        path = Path(video_path).expanduser()
        st.session_state.reader = VideoReader(path)
        st.session_state.video_path = str(path)
        st.success(f"Loaded: {path}")
    except Exception as exc:  # noqa: BLE001
        st.session_state.reader = None
        st.error(f"Failed to load video: {exc}")

reader: VideoReader | None = st.session_state.reader
if reader is not None:
    total_frames = reader.get_frame_count()
    fps = reader.get_fps() if reader.get_fps() > 0 else 1.0

    if total_frames <= 0:
        st.warning("Video appears to have no readable frames.")
    else:
        frame_idx = st.slider("Frame", min_value=0, max_value=total_frames - 1, value=0)
        frame = reader.read_frame(frame_idx)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_sec = frame_idx / fps
        st.caption(f"Frame: {frame_idx} / {total_frames - 1} · Time: {timestamp_sec:.3f}s · FPS: {fps:.3f}")
        st.image(rgb, channels="RGB", use_container_width=True)
