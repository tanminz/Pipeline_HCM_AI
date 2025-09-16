import argparse
import json
import os
from pathlib import Path

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
	parser = argparse.ArgumentParser(description='Create image_path.json from a keyframes directory')
	parser.add_argument('--keyframes_dir', required=True)
	parser.add_argument('--output_json', default='image_path.json')
	parser.add_argument('--relative_to', default='.')
	args = parser.parse_args()

	keyframes_dir = Path(args.keyframes_dir).resolve()
	rel_base = Path(args.relative_to).resolve()

	fps = [p for p in keyframes_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
	if not fps:
		raise SystemExit(f'No images found in {keyframes_dir}')
	fps.sort()  # alphanumeric sort

	mapping = {i: os.path.relpath(str(p), start=str(rel_base)).replace('\\','/') for i, p in enumerate(fps)}

	with open(args.output_json, 'w', encoding='utf-8') as f:
		json.dump(mapping, f, ensure_ascii=False)

	print(f'Wrote {len(mapping)} entries to {args.output_json}')


if __name__ == '__main__':
	main() 