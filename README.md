# LLM2CLIP4CXR

A project for medical image-text retrieval using LLM2CLIP for chest X-ray (CXR) images.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lukeingawesome/llm2clip4cxr.git
cd llm2clip4cxr
```

2. Install the package in development mode:
```bash
pip install -e .
```

This will install all required dependencies from `requirements.txt`.

## Current Status

- ✅ **Retrieval/Inference Code**: Fully implemented and ready to use
- 🚧 **Training Code**: Partially implemented, will be added after sanity checks

## Inference Instructions

### 1. Prepare Your Data

Create CSV files with the following structure:

**Anchor CSV** (e.g., `anchors.csv`):
```csv
img_path,caption_if
/path/to/image1.jpg,"Chest X-ray shows clear lungs"
/path/to/image2.jpg,"Normal cardiac silhouette"
```

**Candidate Text CSV** (e.g., `candidates.csv`):
```csv
caption_if
"Chest X-ray shows clear lungs"
"Normal cardiac silhouette"
"Pneumonia in right lower lobe"
```

**Important Notes:**
- Use **absolute paths** for image files
- The `img_path` column should contain the full path to your images (however the column name does not have to be 'img_path')
- The `caption` column should contain the text descriptions/reports (however the column name does not have to be 'caption')
- **For standard top-k retrieval testing**: The anchor CSV and candidate CSV can be the same file
- **Column validation**: If the specified column names (`--csv-img-key` and `--csv-caption-key`) are not found in your CSV, the system will show a warning and automatically add fake placeholder columns to prevent errors
- **Error handling**: The system will create empty white images (224x224) for fake image paths or when image loading fails, ensuring the pipeline continues without crashes

### 2. Download Model Weights

Download the pre-trained model weights from [Google Drive](https://drive.google.com/file/d/1BzqCtu3X92-AAs03B-16wL5TY91xtAcJ/view?usp=sharing) and place them in your desired location.

### 3. Configure and Run Inference

Edit the `retrieve.sh` script with your specific paths:

```bash
#!/usr/bin/env bash

set -e

CUDA_VISIBLE_DEVICES=0 python -u retrieval.py \
  --anchors /path/to/your/anchors.csv \
  --candidates /path/to/your/candidates.csv \
  --clip-ckpt /path/to/your/model/llm2clip4cxr.bin \
  --text-base lukeingawesome/llm2vec4cxr \
  --csv-img-key img_path \
  --csv-caption-key caption_if \
  --precision bf16 \
  --pooling-mode latent_attention \
  --text-max-len 512 \
  --batch 16 \
  --similarity clip \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --save /path/to/output/retrieval_results.csv \
  --exit-no-segfault \
  --adapter-name llm2clip
```

**Key Parameters to Update:**
- `--anchors`: Path to your anchor CSV file
- `--candidates`: Path to your candidate text CSV file  
- `--clip-ckpt`: Path to your downloaded model weights
- `--csv-img-key`: Column name for image paths in your CSV (default: `img_path`)
- `--csv-caption-key`: Column name for captions in your CSV (default: `caption`)
- `--save`: Output path for retrieval results

### 4. Run the Script

```bash
chmod +x retrieve.sh
./retrieve.sh
```

## Model Configuration

The model uses the following key components:
- **Base Model**: `lukeingawesome/llm2vec4cxr`
- **Precision**: bf16 (for memory efficiency)
- **Pooling**: Latent attention pooling
- **LoRA Configuration**: 
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: q_proj, k_proj, v_proj, o_proj

## Output

The script will generate a CSV file with retrieval results, showing the similarity scores between anchor images and candidate texts.

## Training

Training code is currently under development and will be released after thorough sanity checks. Stay tuned for updates!

## Future Work

- **Python 3.10 Support**: Upcoming updates will ensure full compatibility with Python 3.10
- **SigLIP Integration**: Plans to add SigLIP (Sigmoid Loss for Language-Image Pre-training) options for enhanced performance

## Acknowledgments

This project is built on top of [LLM2CLIP](https://github.com/LLM2CLIP/LLM2CLIP), which provides the foundational architecture for combining Large Language Models with CLIP for vision-language tasks. We extend their work specifically for chest X-ray (CXR) medical image-text retrieval.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper (citation details to be added) and the original LLM2CLIP work:

```bibtex
@article{ko2025exploring,
  title={Exploring the Capabilities of LLM Encoders for Image--Text Retrieval in Chest X-rays},
  author={Ko, Hanbin and Cho, Gihun and Baek, Inhyeok and Kim, Donguk and Koo, Joonbeom and Kim, Changi and Lee, Dongheon and Park, Chang Min},
  journal={arXiv preprint arXiv:2509.15234},
  year={2025}
}
```
