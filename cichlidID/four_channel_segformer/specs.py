import sys, torch, platform

print("=== System ===")
print("Python:", sys.version)
print("OS:", platform.platform())

print("\n=== PyTorch / CUDA ===")
print("PyTorch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n=== Libraries ===")
try:
    import timm
    print("timm:", timm.__version__)
except ImportError:
    print("timm: not found")

print("\n=== DINOv2 Model ===")
try:
    import timm
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=False)
    cfg = model.default_cfg
    print("Model: vit_base_patch14_dinov2")
    print("Input size:", cfg.input_size)
    print("Patch size: 14")
    print("Embed dim:", model.embed_dim)
    print("Depth:", model.depth)
    print("Num heads:", model.num_heads)
except Exception as e:
    print("DINOv2 error:", e)