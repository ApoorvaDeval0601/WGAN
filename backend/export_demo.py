"""
export_demo.py — WGAN Lab 5
Generates a fully self-contained demo.html:
  - Real vs Fake guessing game using CIFAR-10 + trained generator
  - Training loss chart
  - Epoch progression viewer
  - Scoreboard

Run from backend folder:
    python export_demo.py
"""

import os
import json
import base64
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import io

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE_DIR, "model_output")
LOG_FILE   = os.path.join(SAVE_DIR, "training_log.json")
GEN_WEIGHTS= os.path.join(SAVE_DIR, "generator_final.pth")
OUT_FILE   = os.path.join(BASE_DIR, "../demo.html")
NZ             = 100
NUM_GAME_PAIRS = 20    # final rounds in the game
CANDIDATES     = 200   # generate this many, critic picks best 20
CRIT_WEIGHTS   = os.path.join(SAVE_DIR, "critic_final.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

# ── Generator (must match train.py) ──────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ,  512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,   3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.main(x)

# ── Load generator ────────────────────────────────
if not os.path.exists(GEN_WEIGHTS):
    print("✗ generator_final.pth not found. Run train.py first.")
    exit(1)

netG = Generator().to(device)
netG.load_state_dict(torch.load(GEN_WEIGHTS, map_location=device))
netG.eval()
print("✓ Generator loaded")

# ── Critic (must match train.py) ─────────────────
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,   128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,   1, 4, 1, 0, bias=False)
        )
    def forward(self, x): return self.main(x).view(-1)

netC = None
if os.path.exists(CRIT_WEIGHTS):
    netC = Critic().to(device)
    netC.load_state_dict(torch.load(CRIT_WEIGHTS, map_location=device))
    netC.eval()
    print("✓ Critic loaded — using critic scoring to pick best fakes")
else:
    print("⚠ Critic weights not found — picking fakes randomly")

# ── Load training log ─────────────────────────────
if not os.path.exists(LOG_FILE):
    print("✗ training_log.json not found.")
    exit(1)
with open(LOG_FILE) as f:
    log = json.load(f)
print(f"✓ Log loaded: {len(log)} epochs")

# ── Load epoch images ─────────────────────────────
epoch_images_b64 = {}
for entry in log:
    ep   = entry["epoch"]
    path = os.path.join(SAVE_DIR, f"epoch_{ep:03d}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            epoch_images_b64[ep] = base64.b64encode(f.read()).decode()
print(f"✓ Epoch images loaded: {len(epoch_images_b64)}")

# ── Generate fake images for the game ────────────
def tensor_to_b64(t):
    """Convert a (3,32,32) tensor in [-1,1] to base64 PNG, upscaled to 128px."""
    t = (t * 0.5 + 0.5).clamp(0, 1)
    arr = (t.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    pil = Image.fromarray(arr, "RGB").resize((128, 128), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

with torch.no_grad():
    # Generate a large pool of candidates
    noise_pool    = torch.randn(CANDIDATES, NZ, 1, 1, device=device)
    fake_pool     = netG(noise_pool)  # (CANDIDATES, 3, 32, 32)

    if netC is not None:
        # Score every candidate with the critic — higher = more realistic
        scores = netC(fake_pool)                          # (CANDIDATES,)
        top_idx = scores.argsort(descending=True)[:NUM_GAME_PAIRS]
        fake_tensors = fake_pool[top_idx]
        print(f"✓ Critic scored {CANDIDATES} fakes, kept top {NUM_GAME_PAIRS}")
    else:
        fake_tensors = fake_pool[:NUM_GAME_PAIRS]

fake_b64 = [tensor_to_b64(fake_tensors[i]) for i in range(NUM_GAME_PAIRS)]
print(f"✓ Generated {NUM_GAME_PAIRS} fake images")

# ── Load real CIFAR-10 images for the game ────────
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dataset   = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
indices   = torch.randperm(len(dataset))[:NUM_GAME_PAIRS]
real_b64  = []
for idx in indices:
    img_tensor, _ = dataset[idx.item()]
    real_b64.append(tensor_to_b64(img_tensor))
print(f"✓ Loaded {NUM_GAME_PAIRS} real CIFAR-10 images")

# ── Build game rounds: each round has 1 real + 1 fake, shuffled ──
import random
rounds = []
for i in range(NUM_GAME_PAIRS):
    pair   = [
        {"img": real_b64[i],  "label": "real"},
        {"img": fake_b64[i],  "label": "fake"}
    ]
    random.shuffle(pair)
    rounds.append(pair)

rounds_json      = json.dumps(rounds)
log_json         = json.dumps(log)
epoch_images_json= json.dumps({str(k): v for k, v in epoch_images_b64.items()})
total_epochs     = len(log)
last_lossC       = log[-1]["avg_loss_C"]
last_lossG       = log[-1]["avg_loss_G"]

# ── Write HTML ────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>WGAN Lab 5 — Real or Fake? | NMIMS MPSTME</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0d0d14;--card:#1a1a2e;--border:#2a2a45;
  --accent:#7c6bff;--accent2:#ff6b9d;
  --text:#e2e2f0;--muted:#8888aa;
  --green:#4ade80;--red:#f87171;--yellow:#fbbf24;
  --radius:12px;
}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;padding-bottom:4rem}}

/* Header */
header{{background:linear-gradient(135deg,#1a1a40,#0d0d20);border-bottom:1px solid var(--border);padding:1.2rem 2rem;display:flex;align-items:center;gap:1rem}}
.logo{{width:40px;height:40px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex-shrink:0}}
header h1{{font-size:1.2rem;font-weight:700}}
header p{{font-size:0.78rem;color:var(--muted);margin-top:0.15rem}}
.header-stats{{margin-left:auto;display:flex;gap:0.8rem}}
.hstat{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:0.3rem 0.8rem;font-size:0.75rem;color:var(--muted);text-align:center}}
.hstat strong{{display:block;color:var(--accent);font-size:0.95rem}}

/* Tabs */
.tabs{{display:flex;gap:0;border-bottom:1px solid var(--border);background:#0a0a12;padding:0 2rem}}
.tab{{padding:0.8rem 1.4rem;font-size:0.85rem;font-weight:600;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all 0.2s}}
.tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}
.tab:hover{{color:var(--text)}}
.tab-content{{display:none;max-width:1100px;margin:0 auto;padding:2rem 1.5rem}}
.tab-content.active{{display:block}}

/* Game */
.game-header{{text-align:center;margin-bottom:2rem}}
.game-header h2{{font-size:1.4rem;font-weight:700;margin-bottom:0.4rem}}
.game-header p{{color:var(--muted);font-size:0.88rem}}
.round-indicator{{display:flex;justify-content:center;gap:6px;margin:1.2rem 0}}
.round-dot{{width:10px;height:10px;border-radius:50%;background:var(--border);transition:background 0.3s}}
.round-dot.done-correct{{background:var(--green)}}
.round-dot.done-wrong{{background:var(--red)}}
.round-dot.current{{background:var(--accent);box-shadow:0 0 8px var(--accent)}}

.score-bar{{display:flex;justify-content:center;gap:2rem;margin-bottom:2rem}}
.score-item{{text-align:center}}
.score-item .val{{font-size:2rem;font-weight:800}}
.score-item .val.green{{color:var(--green)}}
.score-item .val.red{{color:var(--red)}}
.score-item .val.yellow{{color:var(--yellow)}}
.score-item .lbl{{font-size:0.75rem;color:var(--muted);margin-top:0.2rem}}

.question{{text-align:center;margin-bottom:1.5rem}}
.question h3{{font-size:1rem;color:var(--muted);font-weight:500}}
.question span{{color:var(--text);font-weight:700}}

.cards-row{{display:flex;gap:2rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.5rem}}
.img-card{{
  background:var(--card);border:2px solid var(--border);border-radius:16px;
  padding:1rem;cursor:pointer;transition:all 0.2s;text-align:center;
  width:200px;
}}
.img-card:hover{{border-color:var(--accent);transform:translateY(-4px);box-shadow:0 8px 30px rgba(124,107,255,0.2)}}
.img-card.selected-correct{{border-color:var(--green);background:#0d2a0d;box-shadow:0 0 20px rgba(74,222,128,0.3)}}
.img-card.selected-wrong{{border-color:var(--red);background:#2a0d0d;box-shadow:0 0 20px rgba(248,113,113,0.3)}}
.img-card.reveal-other{{border-color:var(--border);opacity:0.6}}
.img-card img{{width:128px;height:128px;image-rendering:pixelated;border-radius:8px;display:block;margin:0 auto 0.8rem}}
.img-card .card-label{{font-size:0.85rem;font-weight:600;color:var(--muted)}}
.img-card .reveal-tag{{
  display:none;margin-top:0.5rem;padding:0.25rem 0.7rem;
  border-radius:20px;font-size:0.78rem;font-weight:700;
}}
.img-card.selected-correct .reveal-tag,
.img-card.selected-wrong .reveal-tag,
.img-card.reveal-other .reveal-tag{{display:inline-block}}
.tag-real{{background:#1a3a1a;color:var(--green)}}
.tag-fake{{background:#3a1a1a;color:var(--red)}}

.feedback{{text-align:center;margin-bottom:1.5rem;min-height:2rem}}
.feedback .msg{{font-size:1.1rem;font-weight:700;padding:0.5rem 1.5rem;border-radius:8px;display:inline-block}}
.msg.correct{{background:#0d2a0d;color:var(--green)}}
.msg.wrong{{background:#2a0d0d;color:var(--red)}}

.btn{{
  display:inline-flex;align-items:center;gap:0.5rem;
  background:linear-gradient(135deg,var(--accent),#5b4dcc);
  color:#fff;border:none;border-radius:8px;
  padding:0.7rem 1.8rem;font-size:0.9rem;font-weight:600;
  cursor:pointer;transition:opacity 0.2s,transform 0.1s;
}}
.btn:hover{{opacity:0.88}}
.btn:active{{transform:scale(0.97)}}
.btn:disabled{{opacity:0.3;cursor:not-allowed}}
.btn-center{{text-align:center;margin-top:0.5rem}}

/* Result screen */
.result-screen{{display:none;text-align:center;padding:2rem}}
.result-screen.show{{display:block}}
.result-circle{{
  width:120px;height:120px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  margin:0 auto 1.5rem;display:flex;align-items:center;justify-content:center;
  font-size:2.5rem;font-weight:800;color:#fff;
  box-shadow:0 0 40px rgba(124,107,255,0.4);
}}
.result-screen h2{{font-size:1.6rem;font-weight:800;margin-bottom:0.5rem}}
.result-screen p{{color:var(--muted);font-size:0.9rem;margin-bottom:2rem}}
.result-grade{{font-size:1.1rem;color:var(--accent);font-weight:700;margin-bottom:1.5rem}}

/* Stats tab */
.concept-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin-bottom:1.5rem}}
.concept-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;border-top:3px solid var(--accent)}}
.concept-card.pink{{border-top-color:var(--accent2)}}
.concept-card.green{{border-top-color:var(--green)}}
.concept-card h3{{font-size:0.9rem;margin-bottom:0.5rem}}
.concept-card p{{font-size:0.8rem;color:var(--muted);line-height:1.6}}
.stat-row{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem}}
.stat-box{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.4rem;flex:1;min-width:120px}}
.stat-box .val{{font-size:1.5rem;font-weight:700;color:var(--accent)}}
.stat-box .lbl{{font-size:0.75rem;color:var(--muted);margin-top:0.2rem}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}}
@media(max-width:768px){{.grid-2{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.4rem}}
.card-title{{font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);margin-bottom:1rem}}
canvas{{width:100%!important;border-radius:8px}}
.slider-row{{display:flex;align-items:center;gap:1rem;margin-bottom:1rem}}
input[type=range]{{flex:1;accent-color:var(--accent);height:4px}}
.epoch-badge{{background:var(--accent);color:#fff;padding:0.25rem 0.7rem;border-radius:20px;font-size:0.8rem;font-weight:700;min-width:70px;text-align:center}}
.img-container{{background:#08080f;border-radius:8px;display:flex;align-items:center;justify-content:center;min-height:220px;overflow:hidden}}
.img-container img{{max-width:100%;image-rendering:pixelated;border-radius:6px}}
.placeholder{{color:var(--muted);font-size:0.85rem;text-align:center;padding:2rem}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem}}
th,td{{padding:0.65rem 0.9rem;text-align:left;border-bottom:1px solid var(--border)}}
th{{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:0.72rem}}
tr:last-child td{{border-bottom:none}}
.tag{{display:inline-block;padding:0.2rem 0.55rem;border-radius:4px;font-size:0.75rem;font-weight:600}}
.tag.good{{background:#1a3a1a;color:var(--green)}}
.tag.bad{{background:#3a1a1a;color:var(--red)}}
</style>
</head>
<body>

<header>
  <div class="logo">⚡</div>
  <div>
    <h1>Wasserstein GAN — Lab 5</h1>
    <p>ATML · NMIMS MPSTME · CIFAR-10 · PyTorch</p>
  </div>
  <div class="header-stats">
    <div class="hstat"><strong>{total_epochs}</strong>Epochs</div>
    <div class="hstat"><strong>{last_lossC}</strong>Critic Loss</div>
    <div class="hstat"><strong>{last_lossG}</strong>Gen Loss</div>
    <div class="hstat"><strong>RTX 5060 Ti</strong>Trained On</div>
  </div>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('game')">🎮 Real or Fake?</div>
  <div class="tab" onclick="switchTab('stats')">📊 Training Stats</div>
  <div class="tab" onclick="switchTab('compare')">⚖️ GAN vs WGAN</div>
</div>

<!-- ═══════════ GAME TAB ═══════════ -->
<div class="tab-content active" id="tab-game">

  <div id="game-area">
    <div class="game-header">
      <h2>Can you tell Real from Fake?</h2>
      <p>Each round shows 2 images — one real CIFAR-10 photo, one generated by the WGAN. Click which one you think is <strong>real</strong>.</p>
    </div>

    <div class="score-bar">
      <div class="score-item"><div class="val green" id="score-correct">0</div><div class="lbl">✓ Correct</div></div>
      <div class="score-item"><div class="val red"   id="score-wrong">0</div><div class="lbl">✗ Wrong</div></div>
      <div class="score-item"><div class="val yellow" id="score-round">1 / {NUM_GAME_PAIRS}</div><div class="lbl">Round</div></div>
    </div>

    <div class="round-indicator" id="round-dots"></div>

    <div class="question"><h3>Which image is <span>REAL</span>?</h3></div>

    <div class="cards-row" id="cards-row"></div>

    <div class="feedback" id="feedback"></div>

    <div class="btn-center">
      <button class="btn" id="next-btn" onclick="nextRound()" disabled>Next Round →</button>
    </div>
  </div>

  <!-- Result screen -->
  <div class="result-screen" id="result-screen">
    <div class="result-circle" id="result-score-circle"></div>
    <h2 id="result-title"></h2>
    <div class="result-grade" id="result-grade"></div>
    <p id="result-desc"></p>
    <br/>
    <button class="btn" onclick="restartGame()">🔄 Play Again</button>
  </div>

</div>

<!-- ═══════════ STATS TAB ═══════════ -->
<div class="tab-content" id="tab-stats">
  <div class="concept-grid">
    <div class="concept-card">
      <h3>🎯 Wasserstein Distance</h3>
      <p>Measures the "earth mover's distance" between real and fake distributions. Provides smoother gradients even when distributions don't overlap — fixes vanishing gradients.</p>
    </div>
    <div class="concept-card pink">
      <h3>⚖️ Critic vs Discriminator</h3>
      <p>WGAN's critic outputs an unbounded real number (no sigmoid). It scores how "real" an image is rather than giving a binary probability.</p>
    </div>
    <div class="concept-card green">
      <h3>📎 Weight Clipping</h3>
      <p>Critic weights are clamped to [−0.01, 0.01] after each step to enforce the Lipschitz constraint required by Wasserstein distance theory.</p>
    </div>
    <div class="concept-card">
      <h3>🔁 5:1 Training Ratio</h3>
      <p>The critic is trained 5 times per generator step. A well-trained critic gives more informative gradients, making generator training stable and meaningful.</p>
    </div>
  </div>

  <div class="stat-row">
    <div class="stat-box"><div class="val">{total_epochs}</div><div class="lbl">Epochs Trained</div></div>
    <div class="stat-box"><div class="val">RTX 5060 Ti</div><div class="lbl">Trained On</div></div>
    <div class="stat-box"><div class="val">{last_lossC}</div><div class="lbl">Avg Critic Loss (last epoch)</div></div>
    <div class="stat-box"><div class="val">{last_lossG}</div><div class="lbl">Avg Generator Loss (last epoch)</div></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Training Loss Curves</div>
      <canvas id="lossChart" height="200"></canvas>
      <p style="font-size:0.75rem;color:var(--muted);margin-top:0.7rem;">
        Critic loss (negative Wasserstein distance) trends toward −∞ then stabilises.
        Generator loss decreases as it fools the critic better.
      </p>
    </div>
    <div class="card">
      <div class="card-title">Generated Images by Epoch</div>
      <div class="slider-row">
        <input type="range" id="epoch-slider" min="0" max="{total_epochs - 1}" value="0" step="1">
        <span class="epoch-badge" id="epoch-label">Epoch 0</span>
      </div>
      <div class="img-container" id="epoch-img-container">
        <div class="placeholder">Move the slider to browse epoch outputs.</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════ COMPARE TAB ═══════════ -->
<div class="tab-content" id="tab-compare">
  <div class="card">
    <div class="card-title">GAN vs WGAN — Key Differences</div>
    <table>
      <thead><tr><th>Aspect</th><th>GAN (Original)</th><th>WGAN</th></tr></thead>
      <tbody>
        <tr><td>Loss function</td><td>Binary cross-entropy (sigmoid)</td><td>Wasserstein distance (linear)</td></tr>
        <tr><td>Discriminator output</td><td>Probability ∈ [0, 1]</td><td>Unbounded real score (critic)</td></tr>
        <tr><td>Gradient vanishing</td><td><span class="tag bad">Common problem</span></td><td><span class="tag good">Solved</span></td></tr>
        <tr><td>Mode collapse</td><td><span class="tag bad">Frequent</span></td><td><span class="tag good">Greatly reduced</span></td></tr>
        <tr><td>Training stability</td><td><span class="tag bad">Fragile</span></td><td><span class="tag good">More stable</span></td></tr>
        <tr><td>Loss correlation with quality</td><td><span class="tag bad">No correlation</span></td><td><span class="tag good">Correlates with image quality</span></td></tr>
        <tr><td>Lipschitz constraint</td><td>Not required</td><td>Weight clipping (±0.01)</td></tr>
        <tr><td>Optimizer</td><td>Adam</td><td>RMSprop</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
// ── Embedded data ──────────────────────────────────
const ROUNDS       = {rounds_json};
const LOG          = {log_json};
const EPOCH_IMAGES = {epoch_images_json};

// ── Tab switching ──────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i) => {{
    const names = ['game','stats','compare'];
    t.classList.toggle('active', names[i] === name);
  }});
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'stats') initChart();
}}

// ── Game state ─────────────────────────────────────
let currentRound = 0;
let correct = 0, wrong = 0;
let answered = false;

function buildDots() {{
  const el = document.getElementById('round-dots');
  el.innerHTML = '';
  ROUNDS.forEach((_, i) => {{
    const d = document.createElement('div');
    d.className = 'round-dot' + (i === 0 ? ' current' : '');
    d.id = 'dot-' + i;
    el.appendChild(d);
  }});
}}

function renderRound() {{
  answered = false;
  document.getElementById('feedback').innerHTML = '';
  document.getElementById('next-btn').disabled = true;

  const pair = ROUNDS[currentRound];
  const row  = document.getElementById('cards-row');
  row.innerHTML = '';

  pair.forEach((item, idx) => {{
    const card = document.createElement('div');
    card.className = 'img-card';
    card.id = 'card-' + idx;
    card.innerHTML = `
      <img src="data:image/png;base64,${{item.img}}" alt="Image ${{idx+1}}"/>
      <div class="card-label">Image ${{idx+1}}</div>
      <div class="reveal-tag ${{item.label === 'real' ? 'tag-real' : 'tag-fake'}}">
        ${{item.label === 'real' ? '✓ REAL' : '✗ FAKE'}}
      </div>`;
    card.onclick = () => guess(idx);
    row.appendChild(card);
  }});

  document.getElementById('score-round').textContent = (currentRound+1) + ' / ' + ROUNDS.length;

  // Update dots
  document.querySelectorAll('.round-dot').forEach((d,i) => {{
    d.className = 'round-dot';
    if (i < currentRound) {{
      // already answered — colour set by guess()
    }} else if (i === currentRound) {{
      d.classList.add('current');
    }}
  }});
}}

function guess(idx) {{
  if (answered) return;
  answered = true;

  const pair    = ROUNDS[currentRound];
  const chosen  = pair[idx];
  const isCorrect = chosen.label === 'real';

  const card0 = document.getElementById('card-0');
  const card1 = document.getElementById('card-1');

  if (isCorrect) {{
    correct++;
    document.getElementById('card-' + idx).className = 'img-card selected-correct';
    document.getElementById('card-' + (1-idx)).className = 'img-card reveal-other';
    document.getElementById('feedback').innerHTML = '<span class="msg correct">✓ Correct! You spotted the real image.</span>';
    document.getElementById('dot-' + currentRound).className = 'round-dot done-correct';
  }} else {{
    wrong++;
    document.getElementById('card-' + idx).className = 'img-card selected-wrong';
    document.getElementById('card-' + (1-idx)).className = 'img-card reveal-other';
    document.getElementById('feedback').innerHTML = '<span class="msg wrong">✗ Wrong! The other one was real.</span>';
    document.getElementById('dot-' + currentRound).className = 'round-dot done-wrong';
  }}

  document.getElementById('score-correct').textContent = correct;
  document.getElementById('score-wrong').textContent   = wrong;
  document.getElementById('next-btn').disabled = false;

  if (currentRound === ROUNDS.length - 1) {{
    document.getElementById('next-btn').textContent = 'See Results 🏆';
  }}
}}

function nextRound() {{
  currentRound++;
  if (currentRound >= ROUNDS.length) {{
    showResult();
    return;
  }}
  renderRound();
}}

function showResult() {{
  document.getElementById('game-area').style.display = 'none';
  const rs   = document.getElementById('result-screen');
  rs.classList.add('show');
  const pct  = Math.round((correct / ROUNDS.length) * 100);
  const circle = document.getElementById('result-score-circle');
  circle.textContent = pct + '%';

  let grade, title, desc;
  if (pct >= 80) {{
    grade = '🏆 Expert Eye';
    title = 'Impressive! The GAN still has some tells.';
    desc  = 'You correctly identified real images most of the time. With more training epochs, even you might be fooled!';
  }} else if (pct >= 60) {{
    grade = '👀 Good Instincts';
    title = 'Not bad — you beat random chance!';
    desc  = 'You caught some fakes, but the generator fooled you a fair bit. This is the goal of GAN training.';
  }} else if (pct >= 40) {{
    grade = '🤖 The GAN Fooled You';
    title = 'The generator did its job!';
    desc  = 'You struggled to tell real from fake — which means the WGAN is learning realistic distributions.';
  }} else {{
    grade = '🎭 Completely Fooled!';
    title = 'The WGAN wins this round!';
    desc  = 'Almost indistinguishable — the generator has learned to produce convincing images.';
  }}

  document.getElementById('result-title').textContent = title;
  document.getElementById('result-grade').textContent = grade;
  document.getElementById('result-desc').textContent  = desc;
}}

function restartGame() {{
  currentRound = 0; correct = 0; wrong = 0; answered = false;
  document.getElementById('game-area').style.display = 'block';
  document.getElementById('result-screen').classList.remove('show');
  document.getElementById('score-correct').textContent = '0';
  document.getElementById('score-wrong').textContent   = '0';
  document.getElementById('next-btn').textContent = 'Next Round →';
  buildDots();
  renderRound();
}}

// ── Loss chart (lazy init) ──────────────────────────
let chartInited = false;
function initChart() {{
  if (chartInited) return;
  chartInited = true;
  new Chart(document.getElementById('lossChart').getContext('2d'), {{
    type: 'line',
    data: {{
      labels: LOG.map(e => 'E' + e.epoch),
      datasets: [
        {{ label:'Critic Loss',    data:LOG.map(e=>e.avg_loss_C), borderColor:'#7c6bff', backgroundColor:'rgba(124,107,255,0.08)', borderWidth:2, pointRadius:2, tension:0.3, fill:true }},
        {{ label:'Generator Loss', data:LOG.map(e=>e.avg_loss_G), borderColor:'#ff6b9d', backgroundColor:'rgba(255,107,157,0.08)', borderWidth:2, pointRadius:2, tension:0.3, fill:true }}
      ]
    }},
    options:{{
      responsive:true, animation:false,
      plugins:{{legend:{{labels:{{color:'#8888aa',boxWidth:12,font:{{size:11}}}}}}}},
      scales:{{
        x:{{ticks:{{color:'#555',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'rgba(255,255,255,0.04)'}}}},
        y:{{ticks:{{color:'#555',font:{{size:10}}}},grid:{{color:'rgba(255,255,255,0.04)'}}}}
      }}
    }}
  }});

  const slider = document.getElementById('epoch-slider');
  const label  = document.getElementById('epoch-label');
  const box    = document.getElementById('epoch-img-container');

  function showEpoch(ep) {{
    label.textContent = 'Epoch ' + ep;
    const b64 = EPOCH_IMAGES[String(ep)];
    if (b64) {{
      box.innerHTML = '<img src="data:image/png;base64,' + b64 + '" alt="Epoch ' + ep + '"/>';
    }} else {{
      box.innerHTML = '<div class="placeholder">No image for epoch ' + ep + '.</div>';
    }}
  }}
  slider.addEventListener('input', () => showEpoch(slider.value));
  showEpoch(0);
}}

// ── Init ───────────────────────────────────────────
buildDots();
renderRound();
</script>
</body>
</html>"""

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(OUT_FILE) / 1e6
print(f"\\n✓ demo.html created ({size_mb:.1f} MB)")
print(f"  → {os.path.abspath(OUT_FILE)}")
print("  Open demo.html in any browser. No server needed.")