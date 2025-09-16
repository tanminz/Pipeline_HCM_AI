import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
	import faiss
except Exception:
	faiss = None


def load_image_paths(image_path_json: str):
	with open(image_path_json, 'r', encoding='utf-8') as f:
		id2path = json.load(f)
	ordered = [id2path[str(i)] if str(i) in id2path else id2path[i] for i in sorted(map(int, id2path.keys()))]
	return ordered


def load_feature_file(fpath: Path) -> np.ndarray:
	if fpath.suffix.lower() == '.npy':
		arr = np.load(str(fpath))
		return np.asarray(arr, dtype=np.float32)
	elif fpath.suffix.lower() == '.npz':
		d = np.load(str(fpath))
		if 'features' in d:
			arr = d['features']
		else:
			keys = list(d.keys())
			arr = d[keys[0]]
		return np.asarray(arr, dtype=np.float32)
	else:
		raise ValueError(f'Unsupported feature file extension: {fpath.suffix}')


def index_features_recursive(features_dir: Path, feature_exts=('.npy', '.npz')):
	stem_to_path = {}
	for ext in feature_exts:
		for fp in features_dir.rglob(f'*{ext}'):
			stem = fp.stem
			if stem not in stem_to_path:
				stem_to_path[stem] = fp
	return stem_to_path


def stack_rows_replace_zeros(features_list, total_n: int = None):
	valid = [v for v in features_list if v is not None]
	if not valid:
		raise SystemExit('No valid feature vectors found.')
	dim = valid[0].shape[-1]
	rows = []
	for v in features_list:
		if v is None:
			rows.append(np.zeros((dim,), dtype=np.float32))
		else:
			vv = v.astype(np.float32)
			if vv.ndim == 2 and vv.shape[0] == 1:
				vv = vv[0]
			rows.append(vv)
	arr = np.vstack([r[np.newaxis, :] if r.ndim == 1 else r for r in rows])
	if total_n is not None and arr.shape[0] != total_n:
		if arr.shape[0] < total_n:
			pad = np.zeros((total_n - arr.shape[0], dim), dtype=np.float32)
			arr = np.vstack([arr, pad])
		else:
			arr = arr[:total_n]
	return arr


def load_features_video_mode(image_paths, features_dir: Path, feature_exts=('.npy', '.npz')):
	"""Handle case where each .npy file contains multiple feature vectors per video"""
	stem_to_path = index_features_recursive(features_dir, feature_exts)
	
	# Group image paths by video_id (extract from filename like 'L25_V088/246.jpg' -> 'L25_V088')
	video_groups = {}
	for img_path in image_paths:
		path_parts = Path(img_path).parts
		if len(path_parts) >= 2:
			for part in path_parts:
				if '_' in part and ('L' in part or 'V' in part):
					video_id = part
					if video_id not in video_groups:
						video_groups[video_id] = []
					video_groups[video_id].append(img_path)
					break
	
	print(f"Found video groups: {list(video_groups.keys())[:10]}... (total: {len(video_groups)})")
	
	# Map video groups to feature files (try different naming patterns)
	video_to_features = {}
	for video_id in video_groups.keys():
		for stem, fpath in stem_to_path.items():
			if video_id.lower() in stem.lower() or stem.lower() in video_id.lower():
				video_to_features[video_id] = fpath
				break
	
	if not video_to_features:
		feature_files = sorted(stem_to_path.values())
		for i, video_id in enumerate(video_groups.keys()):
			if i < len(feature_files):
				video_to_features[video_id] = feature_files[i]
	
	print(f"Mapped {len(video_to_features)} video groups to feature files")
	
	# Cache features per video to avoid reloading repeatedly
	feature_cache = {}
	
	# Build features array
	all_features = []
	for img_path in image_paths:
		path_parts = Path(img_path).parts
		video_id = None
		frame_idx = 0
		for part in path_parts:
			if '_' in part and ('L' in part or 'V' in part):
				video_id = part
				break
		filename = Path(img_path).name
		try:
			frame_idx = int(filename.split('.')[0])
		except ValueError:
			frame_idx = 0
		
		if video_id in video_to_features:
			feature_file = video_to_features[video_id]
			if video_id not in feature_cache:
				feature_cache[video_id] = load_feature_file(feature_file)
			features = feature_cache[video_id]
			if features.ndim == 2 and features.shape[0] > 0:
				idx = min(max(frame_idx, 0), features.shape[0] - 1)
				frame_features = features[idx]
			else:
				frame_features = features[0] if features.ndim == 2 else features
			all_features.append(frame_features.astype(np.float32))
		else:
			all_features.append(None)
	
	return stack_rows_replace_zeros(all_features, total_n=len(image_paths))


def load_features_basename(image_paths, features_dir: Path, feature_exts=('.npy', '.npz'), pad_missing=False):
	stem_map = index_features_recursive(features_dir, feature_exts)
	features = []
	missing = []
	for p in image_paths:
		stem = Path(p).stem
		fp = stem_map.get(stem)
		if fp is None:
			missing.append(stem)
			features.append(None if pad_missing else None)
			continue
		vec = load_feature_file(fp)
		if vec.ndim == 2 and vec.shape[0] == 1:
			vec = vec[0]
		features.append(np.asarray(vec, dtype=np.float32))
	if missing:
		print(f"[WARN] Missing {len(missing)} features by basename, e.g.: {missing[:5]}")
	if not any(v is not None for v in features):
		raise SystemExit('Basename matching found 0 features. Consider using --match_mode order or --single_file.')
	return stack_rows_replace_zeros(features, total_n=(len(image_paths) if pad_missing else None))


def load_features_order(image_paths, features_dir: Path, feature_glob='*.npy', pad_missing=False):
	fps = sorted(features_dir.rglob(feature_glob))
	if not fps:
		raise SystemExit(f'No feature files found in {features_dir} matching {feature_glob}')
	rows = []
	for fp in fps:
		vec = load_feature_file(fp)
		if vec.ndim == 2:
			if vec.shape[0] != 1:
				raise SystemExit(f'Unexpected 2D feature with rows>1 in {fp}')
			vec = vec[0]
		rows.append(np.asarray(vec, dtype=np.float32))
	if not rows:
		raise SystemExit('No valid feature vectors found in order mode.')
	dim = rows[0].shape[-1]
	mat = np.vstack([r[np.newaxis, :] if r.ndim == 1 else r for r in rows])
	if pad_missing:
		mat = stack_rows_replace_zeros(rows, total_n=len(image_paths))
	else:
		if mat.shape[0] != len(image_paths):
			raise SystemExit(f'Feature count {mat.shape[0]} does not match image count {len(image_paths)}')
	return mat


def load_features_single(single_file: Path, expected_n: int):
	arr = load_feature_file(single_file)
	if arr.ndim != 2:
		raise SystemExit(f'Single feature file must be 2D (N,D), got shape {arr.shape}')
	if arr.shape[0] != expected_n:
		raise SystemExit(f'Feature rows {arr.shape[0]} != expected images {expected_n}')
	return np.asarray(arr, dtype=np.float32)


def normalize_rows(x: np.ndarray):
	norms = np.linalg.norm(x, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	return x / norms


def build_faiss_index(vectors: np.ndarray, out_path: str):
	if faiss is None:
		raise RuntimeError('faiss-cpu not installed. Install with: pip install faiss-cpu')
	dim = vectors.shape[1]
	index = faiss.IndexFlatIP(dim)
	index.add(vectors)
	faiss.write_index(index, out_path)
	print(f'Wrote index with {index.ntotal} vectors to {out_path}')


def main():
	parser = argparse.ArgumentParser(description='Build FAISS index from precomputed CLIP features aligned to image_path.json')
	parser.add_argument('--image_path_json', required=True)
	parser.add_argument('--features_dir', help='Directory containing per-image feature files')
	parser.add_argument('--out_index', default='faiss_from_features.bin')
	parser.add_argument('--out_npy', default=None, help='Optional: path to save aligned features matrix (N,D) as .npy')
	parser.add_argument('--match_mode', choices=['basename','order','single','video_features'], default='video_features', help='How to align features to images')
	parser.add_argument('--feature_glob', default='*.npy', help='Glob for order mode (e.g., *.npy or *.npz)')
	parser.add_argument('--single_file', help='Path to a single 2D feature file (npy/npz) for single mode')
	parser.add_argument('--pad_missing', action='store_true', help='Pad zeros if features fewer than images')
	args = parser.parse_args()

	image_paths = load_image_paths(args.image_path_json)
	print(f'Loaded {len(image_paths)} image paths')

	if args.match_mode == 'video_features':
		if not args.features_dir:
			raise SystemExit('--features_dir is required for video_features mode')
		features = load_features_video_mode(image_paths, Path(args.features_dir))
	elif args.match_mode == 'basename':
		if not args.features_dir:
			raise SystemExit('--features_dir is required for basename mode')
		features = load_features_basename(image_paths, Path(args.features_dir), pad_missing=args.pad_missing)
	elif args.match_mode == 'order':
		if not args.features_dir:
			raise SystemExit('--features_dir is required for order mode')
		features = load_features_order(image_paths, Path(args.features_dir), args.feature_glob, pad_missing=args.pad_missing)
	elif args.match_mode == 'single':
		if not args.single_file:
			raise SystemExit('--single_file is required for single mode')
		features = load_features_single(Path(args.single_file), len(image_paths))
	else:
		raise SystemExit('Unknown match_mode')

	features = normalize_rows(features.astype(np.float32))
	print('Features shape:', features.shape)

	# Save NPY if requested
	if args.out_npy:
		np.save(args.out_npy, features)
		print(f'Saved aligned features to {args.out_npy}')

	# Try FAISS if available
	if faiss is not None:
		try:
			build_faiss_index(features, args.out_index)
		except Exception as e:
			print('[WARN] Could not write FAISS index:', e)
	else:
		print('[INFO] faiss-cpu not installed; skipped writing FAISS index. Use --out_npy for NumPy search.')


if __name__ == '__main__':
	main() 