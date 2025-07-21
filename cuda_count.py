import torch

device_count = torch.cuda.device_count()
print(f"可用 GPU 数量: {device_count}")
for i in range(device_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
