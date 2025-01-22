import os
import torch

try:
    safedriver_path = os.environ["SAFEDRIVER_ACTIONHEAD_PATH"]
    print(safedriver_path)
    safedriver_module = torch.jit.load(safedriver_path)
    safedriver_module.eval()
except Exception as e:
    print("Warning: Could not load SafeDriver model")
    safedriver_module = None
