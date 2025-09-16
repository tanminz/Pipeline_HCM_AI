import os
import numpy as np

try:
	import clip
except Exception:
	clip = None

import torch

class SimpleIndex:
	def __init__(self, npy_file: str, id2img_fps, device='cpu', clip_backbone='ViT-B/32'):
		self.features = np.load(npy_file).astype(np.float32)
		# rows are L2-normalized already ideally; enforce
		norms = np.linalg.norm(self.features, axis=1, keepdims=True)
		norms[norms == 0] = 1.0
		self.features = self.features / norms
		self.id2img_fps = id2img_fps
		self.device = device
		if clip is None:
			raise RuntimeError('CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git')
		self.model, _ = clip.load(clip_backbone, device=device)
		self.model.eval()

	def _cosine_search(self, query_vecs: np.ndarray, k: int):
		# query_vecs (M,D) assumed normalized
		scores = query_vecs @ self.features.T
		idx = np.argsort(-scores, axis=1)[:, :k]
		top = np.take_along_axis(scores, idx, axis=1)
		return top, idx

	def text_search(self, text: str, k: int):
		with torch.no_grad():
			tokens = clip.tokenize([text]).to(self.device)
			text_features = self.model.encode_text(tokens).float().cpu().numpy()
		norms = np.linalg.norm(text_features, axis=1, keepdims=True)
		norms[norms == 0] = 1.0
		text_features = text_features / norms
		scores, idx = self._cosine_search(text_features, k)
		idx = idx.flatten()
		image_paths = [self.id2img_fps[int(i)] for i in idx]
		return scores.flatten(), idx, image_paths

	def image_search(self, id_query: int, k: int):
		query = self.features[id_query:id_query+1]
		scores, idx = self._cosine_search(query, k)
		idx = idx.flatten()
		image_paths = [self.id2img_fps[int(i)] for i in idx]
		return scores.flatten(), idx, image_paths 