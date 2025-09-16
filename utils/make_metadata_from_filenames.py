import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict


def load_media_info(media_info_dir: Path) -> Dict[str, dict]:
	"""Load simple media info from TSV/CSV files in a folder into a dict keyed by video_id."""
	info = {}
	if not media_info_dir.exists():
		return info
	for p in media_info_dir.glob('*.tsv'):
		with open(p, 'r', encoding='utf-8') as f:
			r = csv.DictReader(f, delimiter='\t')
			for row in r:
				vid = row.get('video_id') or row.get('id')
				if vid:
					info[str(vid)] = row
	for p in media_info_dir.glob('*.csv'):
		with open(p, 'r', encoding='utf-8') as f:
			r = csv.DictReader(f)
			for row in r:
				vid = row.get('video_id') or row.get('id')
				if vid:
					info[str(vid)] = row
	return info


def main():
	parser = argparse.ArgumentParser(description='Build minimal metadata JSON from keyframe filenames and optional media-info tables')
	parser.add_argument('--keyframes_dir', required=True)
	parser.add_argument('--media_info_dir', required=False, default='')
	parser.add_argument('--output_json', default='competition_metadata.json')
	parser.add_argument('--relative_to', default='.')
	args = parser.parse_args()

	keyframes_dir = Path(args.keyframes_dir).resolve()
	media_info_dir = Path(args.media_info_dir).resolve() if args.media_info_dir else None
	rel_base = Path(args.relative_to).resolve()

	media_map = load_media_info(media_info_dir) if media_info_dir else {}

	metadata = {}
	for img_fp in keyframes_dir.rglob('*.jpg'):
		rel = os.path.relpath(str(img_fp), start=str(rel_base)).replace('\\','/')
		stem = img_fp.stem
		# Try to parse video_id and frame index from stem (e.g., myvideo_f123)
		video_id = stem
		frame_index = None
		if '_f' in stem:
			parts = stem.rsplit('_f', 1)
			video_id = parts[0]
			try:
				frame_index = int(parts[1])
			except Exception:
				frame_index = None
		mi = media_map.get(video_id, {})
		metadata[rel] = {
			'video_id': video_id,
			'frame_index': frame_index,
			'objects': [],
			'ocr_text': '',
			'place': '',
			'media': mi
		}

	with open(args.output_json, 'w', encoding='utf-8') as f:
		json.dump(metadata, f, ensure_ascii=False)

	print(f'Wrote metadata for {len(metadata)} images to {args.output_json}')


if __name__ == '__main__':
	main() 