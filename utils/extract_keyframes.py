import argparse
import os
from pathlib import Path

try:
	import cv2
except Exception as e:
	raise SystemExit("Please install OpenCV first: pip install opencv-python")


def extract_frames(video_path: Path, out_dir: Path, every_seconds: float = 1.0, max_width: int = 1280):
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		print(f"[WARN] Cannot open video: {video_path}")
		return 0

	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	interval = max(int(round(fps * every_seconds)), 1)
	frame_idx = -1
	saved = 0
	base = video_path.stem

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_idx += 1
		if frame_idx % interval != 0:
			continue
		# resize optionally
		h, w = frame.shape[:2]
		if max_width and w > max_width:
			scale = max_width / float(w)
			frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

		out_fp = out_dir / f"{base}_f{frame_idx}.jpg"
		cv2.imwrite(str(out_fp), frame)
		saved += 1

	cap.release()
	return saved


def main():
	parser = argparse.ArgumentParser(description='Extract keyframes from videos at fixed time interval')
	parser.add_argument('--videos_dir', required=True, help='Directory containing videos')
	parser.add_argument('--out_dir', required=True, help='Output directory for frames')
	parser.add_argument('--every_seconds', type=float, default=1.0, help='Sampling interval in seconds (default: 1.0)')
	parser.add_argument('--max_width', type=int, default=1280, help='Resize width cap (default: 1280). Set 0 to disable')
	args = parser.parse_args()

	videos_dir = Path(args.videos_dir)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}
	video_files = [p for p in videos_dir.rglob('*') if p.suffix.lower() in exts]
	if not video_files:
		raise SystemExit(f'No videos found under: {videos_dir}')

	total_saved = 0
	for vp in video_files:
		saved = extract_frames(vp, out_dir, args.every_seconds, args.max_width)
		print(f"Saved {saved} frames from {vp}")
		total_saved += saved

	print(f"Done. Total frames saved: {total_saved}. Output: {out_dir}")


if __name__ == '__main__':
	main() 