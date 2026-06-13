import kagglehub

# Download latest version
path = kagglehub.dataset_download("vafaeii/kth-action-recognition-dataset")

print("Path to dataset files:", path)