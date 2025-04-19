import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image
import warnings
import os

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
target_class = 0
mutation_rate = 0.05
max_generations = 200000
target_confidence = 0.99
gif_save_interval = 10  # Save image every n generations for GIF

# ----------------------------
# ✅ LOAD DATASET & MEDIAN IMAGE
# ----------------------------
print("📥 Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.uint8)
y = mnist.target.to_numpy().astype(int)
X = X.reshape(-1, 28, 28)

median_image = np.median(X[y == target_class], axis=0)
median_image_norm = median_image / 255.0

# ----------------------------
# ✅ LOAD TRAINED MODEL
# ----------------------------
print("🧠 Loading model...")
model = joblib.load("model_SVM_mnist.pkl")

# ----------------------------
# ✅ INIT EVOLUTION
# ----------------------------
print("🚀 Starting evolution...")
image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
confidences = []
ssims = []
gif_frames = []

os.makedirs("gif_frames", exist_ok=True)

for gen in tqdm(range(max_generations), desc="Evolving"):
    mutated = np.copy(image)

    # Apply random mutations
    num_mutations = int(mutation_rate * image.size)
    for _ in range(num_mutations):
        x, y = np.random.randint(0, 28), np.random.randint(0, 28)
        mutated[x, y] = np.clip(mutated[x, y] + np.random.randint(-10, 11), 0, 255)

    # Evaluate confidence and SSIM
    flat_mutated = mutated.reshape(1, -1)
    confidence = model.predict_proba(flat_mutated)[0][target_class]
    mutated_norm = mutated / 255.0
    ssim_score = ssim(mutated_norm, median_image_norm, data_range=1.0)

    confidences.append(confidence)
    ssims.append(ssim_score)

    # Save frame for GIF
    if gen % gif_save_interval == 0:
        img_pil = Image.fromarray(mutated.astype(np.uint8))
        frame_path = f"gif_frames/frame_{gen:04d}.png"
        img_pil.save(frame_path)
        gif_frames.append(frame_path)

    # Accept new image if confidence improves
    prev_conf = model.predict_proba(image.reshape(1, -1))[0][target_class]
    if confidence > prev_conf:
        image = mutated
    # Early stopping
    if confidence >= target_confidence:
        print(f"🎯 Target confidence {target_confidence} reached at generation {gen}")
        break

# ----------------------------
# ✅ SAVE FINAL RESULTS
# ----------------------------

# Plot Confidence & SSIM
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(confidences, label="Confidence")
plt.xlabel("Generation")
plt.ylabel("Confidence")
plt.title("Confidence over Generations")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ssims, label="SSIM", color="orange")
plt.xlabel("Generation")
plt.ylabel("SSIM")
plt.title("SSIM vs. Median Image")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("evolution_metrics.png")
print("📊 Saved plot to evolution_metrics.png")

# Save final image
plt.figure()
plt.imshow(image, cmap='gray')
plt.title(f"Final Evolved Image\nConf={confidences[-1]:.4f}, SSIM={ssims[-1]:.4f}")
plt.axis("off")
plt.savefig("final_evolved_image.png")
print("🖼️ Saved final image to final_evolved_image.png")

# Generate GIF
frames = [Image.open(fp).convert("P") for fp in gif_frames]
gif_path = "evolution.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
print(f"🎞️ Saved GIF to {gif_path}")
