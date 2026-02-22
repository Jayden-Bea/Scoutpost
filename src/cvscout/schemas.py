from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class Track(BaseModel):
    track_id: int = Field(ge=-1)
    bbox: BBox
    confidence: float = Field(ge=0, le=1, alias="score")

    model_config = {
        "populate_by_name": True,
    }


class FrameTracks(BaseModel):
    frame_idx: int = Field(ge=0)
    timestamp_s: float = Field(default=0, ge=0)
    tracks: list[Track] = Field(default_factory=list)


class StationAssignment(BaseModel):
    frame_idx: int = Field(ge=0)
    station: str
    track_id: int = Field(ge=0)
    team_number: int | None = Field(default=None, ge=0)


class Override(BaseModel):
    frame_idx: int = Field(ge=0)
    type: Literal["manual", "system", "correction"]
    details: dict[str, str | int | float | bool | None]
