from cvscout.schemas import BBox, Track
from cvscout.tracking import iou, track_sequence


def test_iou_identical_boxes() -> None:
    box = BBox(x=10, y=20, width=30, height=40)
    assert iou(box, box) == 1.0


def test_iou_non_overlapping_boxes() -> None:
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=20, y=20, width=10, height=10)
    assert iou(a, b) == 0.0


def test_track_sequence_stable_ids_for_moving_objects() -> None:
    detections_by_frame = [
        [
            Track(track_id=-1, bbox=BBox(x=10, y=10, width=20, height=20), confidence=0.9),
            Track(track_id=-1, bbox=BBox(x=100, y=100, width=20, height=20), confidence=0.9),
        ],
        [
            Track(track_id=-1, bbox=BBox(x=12, y=10, width=20, height=20), confidence=0.88),
            Track(track_id=-1, bbox=BBox(x=102, y=100, width=20, height=20), confidence=0.87),
        ],
        [
            Track(track_id=-1, bbox=BBox(x=14, y=10, width=20, height=20), confidence=0.86),
            Track(track_id=-1, bbox=BBox(x=104, y=100, width=20, height=20), confidence=0.85),
        ],
    ]

    tracked = track_sequence(detections_by_frame, iou_threshold=0.3)

    first_ids = [track.track_id for track in tracked[0].tracks]
    second_ids = [track.track_id for track in tracked[1].tracks]
    third_ids = [track.track_id for track in tracked[2].tracks]

    assert first_ids == [0, 1]
    assert second_ids == [0, 1]
    assert third_ids == [0, 1]
