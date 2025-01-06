import torch
print(torch.__version__)             # Check PyTorch version
print(torch.cuda.is_available())     # Verify CUDA availability
print(torch.cuda.get_device_name(0)) # Show GPU name
