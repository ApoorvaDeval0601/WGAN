"""
export_demo.py
Packages training log + all epoch images into a single self-contained HTML.
Just open demo.html in any browser — no server, no Python needed.

Run from the backend folder:
    python export_demo.py
"""

import os
import json
import base64

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE_DIR, "model_output")
LOG_FILE   = os.path.join(SAVE_DIR, "training_log.json")
OUT_FILE   = os.path.join(BASE_DIR, "../demo.html")

# ── Load training log ──────────────────────────────
if not os.path.exists(LOG_FILE):
    print("✗ training_log.json not found. Run train.py first.")
    exit(1)

with open(LOG_FILE) as f:
    log = json.load(f)

print(f"✓ Loaded log: {len(log)} epochs")

# ── Load epoch images as base64 ────────────────────
images_b64 = {}
for entry in log:
    epoch = entry["epoch"]
    path  = os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            images_b64[epoch] = base64.b64encode(f.read()).decode()

print(f"✓ Loaded {len(images_b64)} epoch images")

# ── Embed everything into HTML ─────────────────────
log_json    = json.dumps(log)
images_json = json.dumps(images_b64)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>WGAN Lab 5 — Demo | NMIMS MPSTME</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #0d0d14; --card: #1a1a2e; --border: #2a2a45;
      --accent: #7c6bff; --accent2: #ff6b9d;
      --text: #e2e2f0; --muted: #8888aa;
      --green: #4ade80; --red: #f87171; --radius: 12px;
    }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; padding-bottom: 3rem; }}
    header {{
      background: linear-gradient(135deg, #1a1a40, #0d0d20);
      border-bottom: 1px solid var(--border);
      padding: 1.5rem 2rem; display: flex; align-items: center; gap: 1.2rem;
    }}
    .logo {{
      width: 42px; height: 42px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 10px; display: flex; align-items: center; justify-content: center;
      font-size: 1.4rem; flex-shrink: 0;
    }}
    header h1 {{ font-size: 1.3rem; font-weight: 700; }}
    header p  {{ font-size: 0.82rem; color: var(--muted); margin-top: 0.2rem; }}
    .badge {{
      margin-left: auto; background: #1a3a1a; border: 1px solid var(--green);
      color: var(--green); border-radius: 20px; padding: 0.35rem 0.9rem;
      font-size: 0.78rem; font-weight: 600;
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
    @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.4rem; }}
    .card-title {{ font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 1rem; }}
    .concept-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
    .concept-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem; border-top: 3px solid var(--accent); }}
    .concept-card.pink  {{ border-top-color: var(--accent2); }}
    .concept-card.green {{ border-top-color: var(--green); }}
    .concept-card h3 {{ font-size: 0.9rem; margin-bottom: 0.5rem; }}
    .concept-card p  {{ font-size: 0.8rem; color: var(--muted); line-height: 1.6; }}
    canvas {{ width: 100% !important; border-radius: 8px; }}
    .slider-row {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }}
    input[type=range] {{ flex: 1; accent-color: var(--accent); height: 4px; }}
    .epoch-badge {{ background: var(--accent); color: #fff; padding: 0.25rem 0.7rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; min-width: 70px; text-align: center; }}
    .img-container {{ background: #08080f; border-radius: 8px; display: flex; align-items: center; justify-content: center; min-height: 220px; overflow: hidden; }}
    .img-container img {{ max-width: 100%; image-rendering: pixelated; border-radius: 6px; }}
    .placeholder {{ color: var(--muted); font-size: 0.85rem; text-align: center; padding: 2rem; }}
    .stat-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
    .stat-box {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.4rem; flex: 1; min-width: 120px; }}
    .stat-box .val {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
    .stat-box .lbl {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 0.65rem 0.9rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.72rem; }}
    tr:last-child td {{ border-bottom: none; }}
    .tag {{ display: inline-block; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
    .tag.good {{ background: #1a3a1a; color: var(--green); }}
    .tag.bad  {{ background: #3a1a1a; color: var(--red); }}
  </style>
</head>
<body>

<header>
  <div class="logo">⚡</div>
  <div>
    <h1>Wasserstein GAN — Lab 5</h1>
    <p>ATML · NMIMS MPSTME · CIFAR-10 · PyTorch</p>
  </div>
  <span class="badge">✓ Offline Demo · {len(log)} Epochs</span>
</header>

<main>
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
    <div class="stat-box"><div class="val" id="stat-epochs">—</div><div class="lbl">Epochs Trained</div></div>
    <div class="stat-box"><div class="val">RTX 5060 Ti</div><div class="lbl">Trained On</div></div>
    <div class="stat-box"><div class="val" id="stat-lossC">—</div><div class="lbl">Avg Critic Loss (last epoch)</div></div>
    <div class="stat-box"><div class="val" id="stat-lossG">—</div><div class="lbl">Avg Generator Loss (last epoch)</div></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Training Loss Curves</div>
      <canvas id="lossChart" height="180"></canvas>
      <p style="font-size:0.75rem;color:var(--muted);margin-top:0.7rem;">
        Critic loss (negative Wasserstein distance) ideally trends toward −∞ then stabilises. Generator loss should decrease as it fools the critic more.
      </p>
    </div>
    <div class="card">
      <div class="card-title">Generated Images by Epoch</div>
      <div class="slider-row">
        <input type="range" id="epoch-slider" min="0" max="{len(log)-1}" value="0" step="1">
        <span class="epoch-badge" id="epoch-label">Epoch 0</span>
      </div>
      <div class="img-container" id="epoch-img-container">
        <div class="placeholder">Move the slider to browse epoch outputs.</div>
      </div>
    </div>
  </div>

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
</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
  const LOG    = {log_json};
  const IMAGES = {images_json};

  // Stats
  const last = LOG[LOG.length - 1];
  document.getElementById('stat-epochs').textContent = LOG.length;
  document.getElementById('stat-lossC').textContent  = last.avg_loss_C.toFixed(4);
  document.getElementById('stat-lossG').textContent  = last.avg_loss_G.toFixed(4);

  // Loss chart
  new Chart(document.getElementById('lossChart').getContext('2d'), {{
    type: 'line',
    data: {{
      labels: LOG.map(e => 'E' + e.epoch),
      datasets: [
        {{ label: 'Critic Loss',    data: LOG.map(e => e.avg_loss_C), borderColor: '#7c6bff', backgroundColor: 'rgba(124,107,255,0.08)', borderWidth: 2, pointRadius: 2, tension: 0.3, fill: true }},
        {{ label: 'Generator Loss', data: LOG.map(e => e.avg_loss_G), borderColor: '#ff6b9d', backgroundColor: 'rgba(255,107,157,0.08)', borderWidth: 2, pointRadius: 2, tension: 0.3, fill: true }}
      ]
    }},
    options: {{
      responsive: true, animation: false,
      plugins: {{ legend: {{ labels: {{ color: '#8888aa', boxWidth: 12, font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#555', maxTicksLimit: 10, font: {{ size: 10 }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#555', font: {{ size: 10 }} }},                    grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
      }}
    }}
  }});

  // Epoch image slider — all images embedded, no server needed
  const slider       = document.getElementById('epoch-slider');
  const epochLabel   = document.getElementById('epoch-label');
  const imgContainer = document.getElementById('epoch-img-container');

  function showEpoch(epoch) {{
    epochLabel.textContent = 'Epoch ' + epoch;
    const b64 = IMAGES[epoch] || IMAGES[String(epoch)];
    if (b64) {{
      imgContainer.innerHTML = '<img src="data:image/png;base64,' + b64 + '" alt="Epoch ' + epoch + '" style="max-width:100%;image-rendering:pixelated;border-radius:6px;"/>';
    }} else {{
      imgContainer.innerHTML = '<div class="placeholder">No image for epoch ' + epoch + '.</div>';
    }}
  }}

  slider.addEventListener('input', () => showEpoch(parseInt(slider.value)));
  showEpoch(0);
</script>
</body>
</html>"""

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(OUT_FILE) / 1e6
print(f"✓ Exported: demo.html ({size_mb:.1f} MB)")
print(f"  → {os.path.abspath(OUT_FILE)}")
print("  Just open demo.html in any browser. No server needed.")
