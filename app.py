from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from cvscout.schemas import FrameTracks
from cvscout.tracking import detect_frame, track_sequence
from cvscout.video import VideoReader


@st.cache_data(show_spinner=False)
def compute_tracks(video_path: str, frame_limit: int) -> list[FrameTracks]:
    with VideoReader(video_path) as reader:
        total = reader.get_frame_count()
        usable_limit = max(0, min(frame_limit, total))
        detections = []
        for frame_idx in range(usable_limit):
            detections.append(detect_frame(reader.read_frame(frame_idx)))
    return track_sequence(detections)


def draw_overlay(frame_bgr, frame_tracks: FrameTracks):
    output = frame_bgr.copy()
    for track in frame_tracks.tracks:
        x1 = int(track.bbox.x)
        y1 = int(track.bbox.y)
        x2 = int(track.bbox.x + track.bbox.width)
        y2 = int(track.bbox.y + track.bbox.height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={track.track_id} conf={track.confidence:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output


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
        show_overlay = st.checkbox("Show tracking overlay", value=False)
        process_limit = st.slider("Process first N frames", min_value=1, max_value=total_frames, value=min(300, total_frames))

        frame = reader.read_frame(frame_idx)

        frame_tracks = FrameTracks(frame_idx=frame_idx, timestamp_s=frame_idx / fps, tracks=[])
        if show_overlay:
            with st.spinner("Running detection + tracking..."):
                tracked_frames = compute_tracks(st.session_state.video_path, process_limit)
            if frame_idx < len(tracked_frames):
                frame_tracks = tracked_frames[frame_idx]
                frame_tracks.timestamp_s = frame_idx / fps
                frame = draw_overlay(frame, frame_tracks)
            else:
                st.info("Current frame is beyond processed range for tracking.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_sec = frame_idx / fps
        st.caption(f"Frame: {frame_idx} / {total_frames - 1} · Time: {timestamp_sec:.3f}s · FPS: {fps:.3f}")
        st.image(rgb, channels="RGB", use_container_width=True)

        if show_overlay:
            active_rows = [
                {
                    "track_id": track.track_id,
                    "bbox": f"({track.bbox.x:.1f}, {track.bbox.y:.1f}, {track.bbox.width:.1f}, {track.bbox.height:.1f})",
                    "score": round(track.confidence, 4),
                }
                for track in frame_tracks.tracks
            ]
            st.subheader("Active tracks")
            st.table(active_rows)
