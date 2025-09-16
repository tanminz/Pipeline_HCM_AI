import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
	import torch
	import clip
except Exception as e:
	raise SystemExit("Please install torch and CLIP: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install git+https://github.com/openai/CLIP.git")

try:
	import faiss
except Exception:
	faiss = None

from PIL import Image


def load_image_paths(image_path_json: str):
	with open(image_path_json, 'r', encoding='utf-8') as f:
		id2path = json.load(f)
	ordered = [id2path[str(i)] if str(i) in id2path else id2path[i] for i in sorted(map(int, id2path.keys()))]
	return ordered


def extract_clip_embeddings(image_paths, device='cpu', clip_backbone='ViT-B/32', batch_size=64):
	model, preprocess = clip.load(clip_backbone, device=device)
	model.eval()
	features = []
	with torch.no_grad():
		for i in range(0, len(image_paths), batch_size):
			batch_paths = image_paths[i:i+batch_size]
			imgs = []
			for p in batch_paths:
				img = Image.open(p).convert('RGB')
				imgs.append(preprocess(img))
			imgs = torch.stack(imgs).to(device)
			f = model.encode_image(imgs).float()
			f = f / f.norm(dim=-1, keepdim=True)
			features.append(f.cpu().numpy().astype(np.float32))
	features = np.concatenate(features, axis=0)
	return features


def build_faiss_index(vectors: np.ndarray, out_path: str):
	if faiss is None:
		raise RuntimeError('faiss-cpu not installed.')
	dim = vectors.shape[1]
	index = faiss.IndexFlatIP(dim)
	index.add(vectors)
	faiss.write_index(index, out_path)
	print(f'Wrote index with {index.ntotal} vectors to {out_path}')


def main():
	parser = argparse.ArgumentParser(description='Build CLIP features from images and optionally FAISS index')
	parser.add_argument('--image_path_json', default='competition_image_paths.json')
	parser.add_argument('--out_npy', default='aligned_features.npy')
	parser.add_argument('--out_index', default='', help='Optional FAISS index path')
	parser.add_argument('--device', default='cpu')
	parser.add_argument('--clip_backbone', default='ViT-B/32')
	parser.add_argument('--batch_size', type=int, default=64)
	args = parser.parse_args()

	paths = load_image_paths(args.image_path_json)
	print(f'Loaded {len(paths)} image paths')

	features = extract_clip_embeddings(paths, device=args.device, clip_backbone=args.clip_backbone, batch_size=args.batch_size)
	print('Features shape:', features.shape)
	np.save(args.out_npy, features)
	print('Saved features to', args.out_npy)

	if args.out_index:
		build_faiss_index(features, args.out_index)


if __name__ == '__main__':
	main()

















