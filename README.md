# Scoutpost

Initial scaffolding for an FRC CV scouting Streamlit app.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Run tests

```bash
uv run pytest
```

## Run app

```bash
uv run streamlit run app.py
```

Then, in the Streamlit UI:
1. Enter a local video path (e.g. `/absolute/path/to/match.mp4`).
2. Click **Load**.
3. Use the frame slider to scrub through the video.

## Project layout

- `app.py` — Streamlit app for loading/scrubbing/displaying frames.
- `src/cvscout/video.py` — OpenCV video reader helper.
- `src/cvscout/schemas.py` — Pydantic schema models.
- `tests/` — pytest coverage for schemas and video frame bounds.
