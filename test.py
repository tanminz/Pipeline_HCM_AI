import torch, cv2, faiss

print("CUDA khả dụng:", torch.cuda.is_available())
print("Tên GPU:", torch.cuda.get_device_name(0))
print("CUDA version build:", torch.version.cuda)

# test faiss
d = 64
index = faiss.IndexFlatL2(d)
print("FAISS index OK, dim =", d)

# test cv2
print("OpenCV version:", cv2.__version__)
