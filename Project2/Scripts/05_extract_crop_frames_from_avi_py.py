from pathlib import Path
import cv2

project_root = Path(r"C:\AI_ML_Fellowship\Project2")

avi_file = project_root / "Video" / "active.avi"
output_dir = project_root / "Images" / "02_Frames"

output_dir.mkdir(parents=True, exist_ok=True)

video = cv2.VideoCapture(str(avi_file))

if not video.isOpened():
    raise RuntimeError(
        f"Cannot open video: {avi_file}"
    )

frame_count = 0

while True:

    success, frame = video.read()

    if not success:
        break

    cropped = frame[
        40:640,   # y : y+600
        0:940     # x : x+940
    ]

    output_file = (
        output_dir /
        f"frame_{frame_count:04d}.png"
    )

    cv2.imwrite(
        str(output_file),
        cropped
    )

    frame_count += 1

video.release()

print(
    f"Completed. "
    f"{frame_count} frames extracted."
)