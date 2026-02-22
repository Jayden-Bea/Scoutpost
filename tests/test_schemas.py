from cvscout.schemas import BBox, FrameTracks, Override, StationAssignment, Track


def test_schemas_validate_and_serialize() -> None:
    frame_tracks = FrameTracks(
        frame_idx=12,
        tracks=[Track(track_id=5, bbox=BBox(x=10, y=5, width=100, height=40), confidence=0.91)],
    )
    station = StationAssignment(frame_idx=12, station="red_1", track_id=5, team_number=254)
    override = Override(frame_idx=12, type="manual", details={"reason": "occlusion", "accepted": True})

    payload = {
        "frame_tracks": frame_tracks.model_dump(),
        "station": station.model_dump(),
        "override": override.model_dump(),
    }

    assert payload["frame_tracks"]["tracks"][0]["bbox"]["width"] == 100
    assert payload["station"]["team_number"] == 254
    assert payload["override"]["details"]["reason"] == "occlusion"
