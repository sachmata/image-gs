# Image-GS Copilot Instructions

This is a research project implementing **Image-GS**: Content-Adaptive Image Representation via 2D Gaussians. It's a novel approach for image compression using differentiable 2D Gaussian splatting.

## Architecture Overview

The core system consists of three main components:

1. **GaussianSplatting2D** (`model.py`): Main optimization class handling 2D Gaussian parameters (position, scale, rotation, features)
2. **Custom CUDA kernels** (`gsplat/`): High-performance differentiable rendering using tile-based rasterization
3. **Image utilities** (`utils/`): Loading, visualization, quantization, and saliency computation

### Key Data Flow
- Input images → 2D Gaussian initialization (gradient/saliency/random) → Progressive optimization → Quantized compression
- Gaussians have 4 parameter types: `xy` (position), `scale`, `rot` (rotation), `feat` (color features)
- Rendering uses either tile-based (`rasterize_gaussians_sum`) or no-tiles (`rasterize_gaussians_no_tiles`) mode

## Essential Development Patterns

### Configuration System
- All parameters defined in `cfgs/default.yaml` 
- Loaded via `utils.misc_utils.load_cfg()` which auto-generates argparse from YAML
- Log directories auto-generated with descriptive names encoding all key parameters

### Progressive Optimization
- Starts with ~50% of target Gaussians (`initial_ratio`)
- Adds Gaussians every `add_steps` based on error maps
- Uses learning rate decay and early stopping when no improvement

### Quantization Strategy
- Parameters quantized using straight-through estimator (`utils.quantization_utils.ste_quantize`)
- Different bit precision for each parameter type: `pos_bits`, `scale_bits`, `rot_bits`, `feat_bits`
- Quantization applied during forward pass, gradients flow through normally

## Key Commands & Workflows

### Basic Image Compression
```bash
# Optimize with 10K Gaussians, half-precision quantization
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize

# Render at higher resolution 
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize --eval --render_height=4000
```

### Texture Stack Compression
```bash
# Process directory of related textures (diffuse, normal, etc.)
python main.py --input_path="textures/alarm-clock_2k" --exp_name="test/alarm-clock_2k" --num_gaussians=30000 --quantize
```

### Build Custom CUDA Extensions
```bash
cd gsplat
pip install -e ".[dev]"
```

## Critical Implementation Details

### CUDA Kernel Constants
- `block_h = block_w = 16`: Tile size (hardcoded in CUDA, modify with extreme caution)
- `topk = 10`: Maximum Gaussians per tile (hardcoded in CUDA kernel)
- `eps = 1e-4` (with tiles) or `1e-7` (no tiles): Numerical stability threshold

### Gaussian Initialization Modes
- `"gradient"`: Sample positions based on image gradient magnitude
- `"saliency"`: Use pre-trained EML-Net saliency model (requires downloading models to `models/emlnet/`)
- `"random"`: Uniform random sampling

### Scale Parameter Logic  
- `disable_inverse_scale=False` (default): Optimize `1/scale` for numerical stability
- Initial scale in pixels, but optimization works in inverse space
- Upsample ratio multiplied to scale during rendering

### Loss Function Combination
- L1 + SSIM by default (`l1_loss_ratio=1.0`, `ssim_loss_ratio=0.1`)
- Uses `fused_ssim` for GPU-accelerated SSIM computation
- L2 loss available but typically disabled

## File Organization Conventions

### Results Structure
```
results/{exp_name}/{auto_generated_config_string}/
├── cfg_train.yaml          # Training configuration snapshot
├── log_train.txt          # Detailed training logs
├── checkpoints/           # Model checkpoints (.pt files)
├── train/                # Intermediate results during training
└── eval/                 # Final evaluation renders
```

### Media Structure  
```
media/
├── images/               # Single images for compression
└── textures/            # Texture stacks (directories with multiple related images)
```

## Dependencies & Environment

### Core Dependencies
- PyTorch 2.4+ with CUDA 12.4
- Custom `gsplat` package (workspace member)
- `fused-ssim` from git (not PyPI)
- Various vision libraries: opencv, matplotlib, scikit-image

### Environment Setup Priority
1. Use provided `environment.yml` for conda setup
2. Fallback to `pyproject.toml` for pip/uv (includes workspace gsplat build)
3. Install gsplat separately: `cd gsplat && pip install -e ".[dev]"`

## Performance Considerations

- **Tile-based rendering** is much faster than no-tiles mode - avoid `--disable_tiles` except for debugging
- **Progressive optimization** significantly improves quality - avoid `--disable_prog_optim` 
- GPU memory usage scales with number of Gaussians and image resolution
- Rendering time: ~0.3K MACs per pixel for hardware-friendly random access

## Common Debugging Patterns

- Check CUDA compilation: gsplat build errors often indicate PyTorch/CUDA version mismatches
- Gradient initialization issues: ensure input images are in `media/` directory
- Quantization artifacts: reduce bit precision gradually to isolate issues
- Memory issues: reduce `num_gaussians` or image resolution

## Integration Points

- **Saliency models**: EML-Net integration for guided initialization
- **Image formats**: Support for JPEG, PNG, TIFF, EXR via OpenCV
- **Metrics**: PSNR, SSIM, LPIPS, FLIP, MS-SSIM evaluation pipeline
- **Visualization**: matplotlib-based Gaussian position/footprint plotting