import argparse
import json
import os
from pathlib import Path

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_images(root_dir: Path):
	for p in root_dir.rglob('*'):
		if p.suffix.lower() in SUPPORTED_EXTS and p.is_file():
			yield p


def main():
	parser = argparse.ArgumentParser(description='Generate image_path.json mapping for the app')
	parser.add_argument('--images_dir', required=True, help='Directory containing images (will be scanned recursively)')
	parser.add_argument('--output_json', default='image_path.json', help='Output JSON file path')
	parser.add_argument('--relative_to', default='.', help='Make paths relative to this directory (default: project root)')
	args = parser.parse_args()

	images_dir = Path(args.images_dir).resolve()
	rel_base = Path(args.relative_to).resolve()

	image_paths = list(find_images(images_dir))
	if not image_paths:
		raise SystemExit(f'No images found under: {images_dir}')

	image_paths = sorted(image_paths)

	id2path = {}
	for idx, p in enumerate(image_paths):
		rel_path = os.path.relpath(str(p), start=str(rel_base))
		rel_path = rel_path.replace('\\', '/')
		id2path[idx] = rel_path

	with open(args.output_json, 'w', encoding='utf-8') as f:
		json.dump(id2path, f, ensure_ascii=False)

	print(f'Wrote {len(id2path)} entries to {args.output_json}')


if __name__ == '__main__':
	main() 