import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
	import faiss
except Exception as e:
	faiss = None

try:
	import torch
	exists_torch = True
except Exception:
	exists_torch = False

try:
	import clip
except Exception as e:
	clip = None

from PIL import Image


def load_image_paths(image_path_json: str):
	with open(image_path_json, 'r', encoding='utf-8') as f:
		id2path = json.load(f)
	# keys may be strings
	ordered = [id2path[str(i)] if str(i) in id2path else id2path[i] for i in sorted(map(int, id2path.keys()))]
	return ordered


def extract_clip_embeddings(image_paths, device='cpu', clip_backbone='ViT-B/32', batch_size=64):
	if clip is None:
		raise RuntimeError('openai-clip library not installed. Install with: pip install git+https://github.com/openai/CLIP.git')
	model, preprocess = clip.load(clip_backbone, device=device)
	features = []
	with torch.no_grad():
		for i in range(0, len(image_paths), batch_size):
			batch_paths = image_paths[i:i+batch_size]
			images = []
			for p in batch_paths:
				img = Image.open(p).convert('RGB')
				images.append(preprocess(img))
			images = torch.stack(images).to(device)
			image_features = model.encode_image(images)
			image_features = image_features / image_features.norm(dim=-1, keepdim=True)
			features.append(image_features.cpu().numpy().astype(np.float32))
	features = np.concatenate(features, axis=0)
	return features


def build_faiss_index(vectors: np.ndarray, out_path: str):
	if faiss is None:
		raise RuntimeError('faiss-cpu not installed. Install with: pip install faiss-cpu')
	dim = vectors.shape[1]
	index = faiss.IndexFlatIP(dim)
	index.add(vectors)
	faiss.write_index(index, out_path)
	print(f'Wrote index with {index.ntotal} vectors to {out_path}')


def main():
	parser = argparse.ArgumentParser(description='Build FAISS index (CLIP) from image_path.json')
	parser.add_argument('--image_path_json', default='image_path.json')
	parser.add_argument('--out_index', default='faiss_normal_ViT.bin')
	parser.add_argument('--clip_backbone', default='ViT-B/32')
	parser.add_argument('--device', default='cpu')
	parser.add_argument('--batch_size', type=int, default=64)
	args = parser.parse_args()

	image_paths = load_image_paths(args.image_path_json)
	print(f'Loaded {len(image_paths)} image paths')

	features = extract_clip_embeddings(image_paths, device=args.device, clip_backbone=args.clip_backbone, batch_size=args.batch_size)
	print('Extracted features:', features.shape)

	build_faiss_index(features, args.out_index)


if __name__ == '__main__':
	main() 