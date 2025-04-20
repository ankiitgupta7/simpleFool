import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import lpips
import torch
import torchvision.transforms as T
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
target_class = 0
mutation_rate = 0.05
max_generations = 200000
target_confidence = 0.9999999999999
gif_save_interval = 10
num_lpips_refs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# ✅ LOAD DATASET & REFERENCES
# ----------------------------
print("📥 Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.uint8)
y = mnist.target.to_numpy().astype(int)
X = X.reshape(-1, 28, 28)

# Select real reference samples from target class
target_images = X[y == target_class]
np.random.seed(42)
reference_samples = target_images[np.random.choice(len(target_images), num_lpips_refs, replace=False)]

# LPIPS setup
lpips_model = lpips.LPIPS(net='alex').to(device)
to_tensor = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])
def to_lpips_tensor(img_np):
    pil = Image.fromarray(img_np)
    return to_tensor(pil).unsqueeze(0).to(device)

reference_tensors = [to_lpips_tensor(img) for img in reference_samples]

# ----------------------------
# ✅ LOAD CLASSIFIER MODEL
# ----------------------------
print("🧠 Loading model...")
model = joblib.load("model_SVM_mnist.pkl")

# ----------------------------
# ✅ INIT EVOLUTION
# ----------------------------
print("🚀 Starting evolution...")
image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
confidences, lpips_scores, gif_frames = [], [], []
os.makedirs("gif_frames", exist_ok=True)
font = ImageFont.load_default()

progress_bar = tqdm(range(max_generations), desc="Evolving", dynamic_ncols=True)

for gen in progress_bar:
    mutated = np.copy(image)
    num_mutations = int(mutation_rate * image.size)

    for _ in range(num_mutations):
        x, y = np.random.randint(0, 28), np.random.randint(0, 28)
        mutated[x, y] = np.clip(mutated[x, y] + np.random.randint(-10, 11), 0, 255)

    flat_mutated = mutated.reshape(1, -1)
    confidence = model.predict_proba(flat_mutated)[0][target_class]

    # LPIPS to references
    mutated_tensor = to_lpips_tensor(mutated)
    lpips_vals = [lpips_model(mutated_tensor, ref).item() for ref in reference_tensors]
    avg_lpips = np.mean(lpips_vals)

    # Fitness = confidence only
    fitness = confidence
    confidences.append(confidence)
    lpips_scores.append(avg_lpips)

    # Update progress bar description
    progress_bar.set_postfix(conf=f"{confidence:.4f}", lpips=f"{avg_lpips:.4f}")

    # Save annotated frame
    if gen % gif_save_interval == 0:
        img_pil = Image.fromarray(mutated.astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        label = f"Gen {gen} | Conf: {confidence:.2f} | LPIPS: {avg_lpips:.2f}"
        draw.text((2, 2), label, font=font, fill=(255, 0, 0))
        frame_path = f"gif_frames/frame_{gen:04d}.png"
        img_pil.save(frame_path)
        gif_frames.append(frame_path)

    # Compare to previous
    prev_conf = model.predict_proba(image.reshape(1, -1))[0][target_class]
    if confidence > prev_conf:
        image = mutated

    if confidence >= target_confidence:
        print(f"🎯 Target confidence {target_confidence} reached at generation {gen}")
        break

# ----------------------------
# ✅ SAVE RESULTS
# ----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(confidences, label="Confidence")
plt.xlabel("Generation")
plt.ylabel("Confidence")
plt.title("Model Confidence")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lpips_scores, label="LPIPS (lower is better)", color="orange")
plt.xlabel("Generation")
plt.ylabel("LPIPS")
plt.title("Perceptual Distance")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("evolution_metrics_lpips.png")
print("📊 Saved plot to evolution_metrics_lpips.png")

plt.figure()
plt.imshow(image, cmap='gray')
plt.title(f"Final Image\nConf={confidences[-1]:.4f}, LPIPS={lpips_scores[-1]:.4f}")
plt.axis("off")
plt.savefig("final_evolved_image_lpips.png")
print("🖼️ Saved final image to final_evolved_image_lpips.png")

# Save GIF
frames = [Image.open(fp).convert("P") for fp in gif_frames]
frames[0].save(
    "evolution_lpips.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
print("🎞️ Saved GIF to evolution_lpips.gif")
