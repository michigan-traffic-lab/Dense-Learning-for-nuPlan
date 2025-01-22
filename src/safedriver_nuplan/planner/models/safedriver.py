import os
import torch

safedriver_path = os.environ["SAFEDRIVER_ACTIONHEAD_PATH"]
print(safedriver_path)
safedriver_module = torch.jit.load(safedriver_path)
safedriver_module.eval()
