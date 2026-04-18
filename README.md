# WGAN — Lab 5 | ATML · NMIMS MPSTME

**Wasserstein Generative Adversarial Network on CIFAR-10**  
Course: Advanced Topics in Machine Learning | Sem VI

---

## What this project is

A complete implementation of WGAN (Arjovsky et al., 2017) on CIFAR-10, with:
- **`backend/train.py`** — training script, GPU-optimised for RTX 3000+ / 5000-series
- **`backend/app.py`** — FastAPI server that serves loss logs, epoch images, and on-demand generation
- **`frontend/index.html`** — single-file dashboard with loss charts, epoch viewer, and live generation

---

## Quick Start (VSCode / Local)

### 1. Install dependencies

```bash
# Create a virtual env (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install backend deps
pip install -r backend/requirements.txt

# PyTorch with CUDA 12.x (for RTX 5060 Ti / any RTX 30–50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Train the model

```bash
cd backend
python train.py
```

- Downloads CIFAR-10 automatically into `./data/`
- Saves a `model_output/epoch_NNN.png` grid every epoch
- Saves `model_output/training_log.json` after every epoch (used by the frontend)
- Saves final weights: `model_output/generator_final.pth` and `critic_final.pth`

Training time on RTX 5060 Ti (16 GB): ~**4–6 minutes per epoch** with batch size 256 + AMP.

### 3. Start the API server

```bash
# In a separate terminal, inside /backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the frontend

```
Double-click frontend/index.html
```
or open `http://localhost:5500` with VSCode Live Server.

The frontend auto-detects it's running locally and points to `http://localhost:8000`.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/status` | Model ready? epochs done? GPU info |
| `GET /api/log` | Full training loss log (JSON) |
| `GET /api/epochs` | List of saved epoch image filenames |
| `GET /api/epoch/{n}` | PNG image grid for epoch n |
| `GET /api/generate?n=16` | Generate n fresh images (base64 PNG) |

---

## Deploying to Vercel (frontend only)

The frontend is a single static HTML file — Vercel deploys it instantly.

```bash
# Install Vercel CLI (optional)
npm i -g vercel
vercel deploy
```

Or just drag-and-drop the `frontend/` folder on vercel.com.

> **Note:** The Python API runs locally. When the frontend is on Vercel and the  
> API is on your local machine, you need to expose it (e.g. via ngrok):  
> `ngrok http 8000`  
> Then update the `API` variable in `frontend/index.html` line 1 of the `<script>` block.

For classmates running locally — they just need to:
1. Install requirements
2. Run `python train.py`
3. Run `uvicorn app:app --reload`
4. Open `frontend/index.html`

---

## GPU Optimisations (RTX 5060 Ti specific)

| Optimisation | Why |
|---|---|
| `torch.backends.cuda.matmul.allow_tf32 = True` | TF32 gives ~3× matmul speedup on Ampere+ with near-identical results |
| `torch.backends.cudnn.benchmark = True` | cuDNN auto-selects fastest conv algorithm for your GPU |
| `batch_size = 256` | 16 GB VRAM handles this easily; keeps GPU saturated |
| `pin_memory=True` + `non_blocking=True` | Overlaps CPU→GPU transfers with compute |
| `zero_grad(set_to_none=True)` | Slightly faster than filling with zeros |
| Mixed Precision (AMP) | `autocast` + `GradScaler` — halves memory, speeds up ~1.5–2× |
| `persistent_workers=True` | Workers stay alive between epochs |

---

## Architecture

```
Generator
  Input:  noise (100, 1, 1)
  ConvTranspose2d → 512×4×4
  ConvTranspose2d → 256×8×8
  ConvTranspose2d → 128×16×16
  ConvTranspose2d →   3×32×32  (Tanh → [-1, 1])

Critic
  Input:  image (3, 32, 32)
  Conv2d → 128×16×16  (LeakyReLU)
  Conv2d → 256×8×8    (BN + LeakyReLU)
  Conv2d → 512×4×4    (BN + LeakyReLU)
  Conv2d →   1×1×1    (linear score, NO sigmoid)
```

---

## Key WGAN Concepts (Lab Questions)

**Wasserstein Distance** — "Earth mover's distance"; measures how much work is needed to transform one distribution into another. Provides meaningful gradients even when real and fake distributions have no overlap.

**Why WGAN > GAN** — Standard GAN uses JS divergence which is constant (= log 2) when distributions don't overlap → zero gradients → training stalls. Wasserstein distance doesn't have this problem.

**Lipschitz Constraint** — The critic must be 1-Lipschitz continuous for the Wasserstein distance formula to be valid. Enforced here by clamping weights to [−0.01, 0.01] after each critic step.

**Critic vs Discriminator** — The critic is not a classifier; it outputs a real-valued score. Higher = more realistic. No sigmoid.

**5:1 ratio** — Critic is trained 5× per generator step. A well-trained critic provides accurate Wasserstein gradients so the generator update is meaningful.

---

## Project Structure

```
wgan-lab5/
├── backend/
│   ├── train.py          ← Run this to train
│   ├── app.py            ← Run this to serve the API
│   └── requirements.txt
├── frontend/
│   └── index.html        ← Open this in a browser
├── model_output/         ← Auto-created during training
│   ├── epoch_000.png
│   ├── epoch_001.png
│   ├── ...
│   ├── training_log.json
│   ├── generator_final.pth
│   └── critic_final.pth
├── vercel.json
└── README.md
```

---

## References

1. Arjovsky et al. (2017) — *Wasserstein GAN* — https://arxiv.org/abs/1701.07875  
2. Goodfellow et al. (2014) — *Generative Adversarial Nets* — https://arxiv.org/abs/1406.2661  
3. Salimans et al. (2016) — *Improved Techniques for Training GANs* — https://arxiv.org/abs/1606.03498
