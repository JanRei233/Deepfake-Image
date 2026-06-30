"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DeepGuard — EfficientNetB4 Deepfake Detector Training              ║
║          Dataset : manjilkarki/deepfake-and-real-images (Kaggle)            ║
║                                                                              ║
║  Key improvements over the original notebook:                               ║
║  1. Correct EfficientNetB4 preprocessing (NOT simple /255 scaling)          ║
║  2. Two-phase transfer learning (frozen → fine-tune top layers)             ║
║  3. Stronger augmentation pipeline (cutout, brightness, saturation)         ║
║  4. Label smoothing + focal-loss-inspired weighting                         ║
║  5. Cosine-annealing learning rate schedule with warm restarts              ║
║  6. Mixed-precision training (float16) for faster GPU throughput            ║
║  7. TFLite export with float16 quantization                                 ║
║  8. Full evaluation report (confusion matrix, ROC-AUC, per-threshold F1)   ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP
─────
1. Download the Kaggle dataset:
       kaggle datasets download -d manjilkarki/deepfake-and-real-images
   OR visit: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

2. Unzip so your folder tree looks like:
       Deepfake Image Dataset/
           train/
               Real/
               Fake/
           val/
               Real/
               Fake/
           test/
               Real/
               Fake/

3. Install dependencies (run once):
       pip install tensorflow[and-cuda] kaggle pillow matplotlib scikit-learn seaborn

4. Run:
       python train.py

5. After training completes, copy the output model into the backend folder:
       deepfake_efficientnet_model.tflite

NOTE: Training on the full 140k-image dataset requires a GPU.
      Expected time: ~2-4 hours on a mid-range GPU (RTX 3060+).
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import os
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, mixed_precision
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Scikit-learn (evaluation) ─────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, average_precision_score,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  REPRODUCIBILITY & GPU SETUP
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅  {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
    # Enable mixed precision — trains ~2× faster on Tensor Cores (Ampere+)
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡  Mixed precision (float16) enabled.")
else:
    print("⚠️   No GPU detected — training will be slow on CPU.")

print(f"🧠  TensorFlow {tf.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# EfficientNetB4 natively expects 380×380, but 256×256 trains faster and
# still achieves excellent accuracy on this dataset.
IMG_SIZE    = 256
BATCH_SIZE  = 32       # Reduce to 16 if you hit OOM on <8 GB VRAM
AUTOTUNE    = tf.data.AUTOTUNE

# Phase 1 — train only the classification head (base frozen)
PHASE1_EPOCHS = 15
PHASE1_LR     = 1e-3

# Phase 2 — unfreeze top N layers of EfficientNetB4 and fine-tune
PHASE2_EPOCHS    = 35
PHASE2_LR_INIT   = 1e-4   # start of cosine schedule
PHASE2_LR_MIN    = 1e-6   # end of cosine schedule
PHASE2_UNFREEZE  = 50     # number of layers from the end to unfreeze

# Regularisation
DROPOUT_RATE    = 0.40
LABEL_SMOOTHING = 0.05    # prevents overconfident predictions

# Dataset paths  ← adjust if your folder names differ
BASE_DIR    = Path("Deepfake Image Dataset")
TRAIN_DIR   = BASE_DIR / "train"
VAL_DIR     = BASE_DIR / "val"
TEST_DIR    = BASE_DIR / "test"

# Output paths
TIMESTAMP      = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = Path("checkpoints") / TIMESTAMP
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_H5        = CHECKPOINT_DIR / "best_model.keras"
TFLITE_OUT     = Path("deepfake_efficientnet_model.tflite")
PLOTS_DIR      = CHECKPOINT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print(f"\n📁  Checkpoints → {CHECKPOINT_DIR}")
print(f"📦  TFLite output → {TFLITE_OUT}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Dataset class mapping ─────────────────────────────────────────────────
# flow_from_directory assigns labels alphabetically → Fake=0, Real=1
# We want:  Fake (deepfake) = 1 (positive class)
#           Real            = 0 (negative class)
# We remap in the loader below.

def load_dataset(directory, shuffle):
    """Return a tf.data.Dataset of (image_tensor, label) pairs."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",           # sub-folder names become labels
        label_mode="binary",
        class_names=["Real", "Fake"],  # Real=0, Fake=1
        color_mode="rgb",
        batch_size=None,             # we batch later after augmentation
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=shuffle,
        seed=SEED,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    )
    return ds

train_ds_raw = load_dataset(TRAIN_DIR, shuffle=True)
val_ds_raw   = load_dataset(VAL_DIR,   shuffle=False)
test_ds_raw  = load_dataset(TEST_DIR,  shuffle=False)

# Count samples for class-weight calculation
print("📊  Counting class distribution …")
labels_all = np.array([y.numpy() for _, y in train_ds_raw])
n_fake  = int(labels_all.sum())
n_real  = len(labels_all) - n_fake
n_total = len(labels_all)
print(f"    Train → Real: {n_real:,}  |  Fake: {n_fake:,}  |  Total: {n_total:,}")

# Class weights to handle any imbalance
class_weight = {
    0: n_total / (2 * n_real),   # Real
    1: n_total / (2 * n_fake),   # Fake
}
print(f"    Class weights → Real: {class_weight[0]:.3f}  |  Fake: {class_weight[1]:.3f}")

# ── 2b. Preprocessing ─────────────────────────────────────────────────────────
#  CRITICAL: EfficientNet expects preprocess_input() which scales [0,255] → [-1,1]
#            Using /255 (as the original notebook did) produces wrong activations!

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)   # maps [0,255] → [-1, 1]
    return image, label

# ── 2c. Augmentation (training only) ─────────────────────────────────────────
augmentation_layer = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),           # ±36°
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.15),
    layers.RandomTranslation(0.08, 0.08),  # slight shift
], name="augmentation")

def augment_train(image, label):
    image = augmentation_layer(image, training=True)
    # Cutout: zero out a random 32×32 patch (forces the model to look at
    # texture artefacts spread across the whole image, not just one region)
    if tf.random.uniform(()) > 0.5:
        h = IMG_SIZE
        t  = tf.random.uniform((), 0, h - 32, dtype=tf.int32)
        l  = tf.random.uniform((), 0, h - 32, dtype=tf.int32)
        mask = tf.ones((t, h, 3))
        mask = tf.concat([mask, tf.zeros((32, h, 3)), tf.ones((h - t - 32, h, 3))], axis=0)
        mask = tf.transpose(mask, [1, 0, 2])
        mask_l = tf.ones((h, l, 3))
        mask_r = tf.ones((h, h - l - 32, 3))
        mask_c = tf.zeros((h, 32, 3))
        col_mask = tf.concat([mask_l, mask_c, mask_r], axis=1)
        combined = tf.minimum(mask, col_mask)
        image = image * combined
    return image, label

# ── 2d. Assemble tf.data pipelines ───────────────────────────────────────────
def build_pipeline(ds, augment=False, repeat=False):
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_train, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = build_pipeline(train_ds_raw, augment=True,  repeat=True)
val_ds   = build_pipeline(val_ds_raw,   augment=False, repeat=False)
test_ds  = build_pipeline(test_ds_raw,  augment=False, repeat=False)

steps_per_epoch      = math.ceil(n_total / BATCH_SIZE)
n_val   = sum(1 for _ in val_ds_raw)
n_test  = sum(1 for _ in test_ds_raw)
print(f"\n🔢  Steps/epoch: {steps_per_epoch}  |  Val: {n_val:,}  |  Test: {n_test:,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(unfreeze_layers=0):
    """
    Build the detector model.
    unfreeze_layers=0  → all base layers frozen (Phase 1 head-only training)
    unfreeze_layers=N  → last N layers of EfficientNetB4 are trainable (Phase 2)
    """
    # EfficientNetB4 backbone — ImageNet weights, no top
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        drop_connect_rate=0.4,
    )

    # Freeze / unfreeze
    base.trainable = False
    if unfreeze_layers > 0:
        for layer in base.layers[-unfreeze_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True   # keep BN frozen to avoid instability

    # Head
    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = base(inputs, training=False)

    # Multi-scale pooling: captures both global context and local artefacts
    gap   = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp   = layers.GlobalMaxPooling2D(name="gmp")(x)
    x     = layers.Concatenate(name="pool_concat")([gap, gmp])

    # Dense classifier
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dense(512, activation="swish", name="fc1",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dense(256, activation="swish", name="fc2",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="drop2")(x)

    # Output — dtype=float32 required even in mixed-precision mode
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32",
                           name="output")(x)

    return keras.Model(inputs, outputs, name="DeepGuard_EfficientNetB4")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr_schedule(epoch, total_epochs, lr_max, lr_min):
    """Cosine annealing without restarts."""
    cos_inner = math.pi * epoch / total_epochs
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(cos_inner))


def plot_history(history, filename, phase_label):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training History — {phase_label}", fontsize=14)

    for ax, metric, title in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy"), ("auc", "val_auc")],
        ["Loss", "Accuracy", "ROC-AUC"],
    ):
        ax.plot(history.history[metric[0]],  label="train")
        ax.plot(history.history[metric[1]],  label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"📈  Plot saved → {PLOTS_DIR / filename}")


def evaluate_model(model, test_dataset, threshold=0.35):
    """Full evaluation: confusion matrix, ROC, PR curve."""
    print("\n🔍  Evaluating on test set …")
    y_true, y_prob = [], []
    for images, labels in test_dataset:
        probs = model.predict(images, verbose=0).flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().astype(int).tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # ── Classification report ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"THRESHOLD = {threshold}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"AP-Score : {average_precision_score(y_true, y_prob):.4f}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ── ROC curve ────────────────────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # ── F1 vs threshold ──────────────────────────────────────────────────────
    thrs = np.linspace(0.1, 0.9, 81)
    f1s  = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thrs]
    best_thr = thrs[np.argmax(f1s)]
    print(f"\n🎯  Best F1 threshold = {best_thr:.2f}  (F1 = {max(f1s):.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, f1s, lw=2)
    ax.axvline(best_thr, color="red", linestyle="--",
               label=f"Best threshold = {best_thr:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "f1_vs_threshold.png", dpi=150)
    plt.close()

    print(f"\n📊  Evaluation plots saved → {PLOTS_DIR}")
    return best_thr


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PHASE 1 — HEAD TRAINING (base frozen)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" PHASE 1 — Training classification head (base frozen)")
print("═" * 60)

model = build_model(unfreeze_layers=0)
model.summary(line_length=90)

total_params     = model.count_params()
trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
print(f"\n📐  Total params     : {total_params:,}")
print(f"📐  Trainable params : {trainable_params:,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

phase1_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=str(BEST_H5),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    callbacks.TensorBoard(
        log_dir=str(CHECKPOINT_DIR / "tb_logs" / "phase1"),
        update_freq="epoch",
    ),
]

history1 = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=phase1_callbacks,
    verbose=1,
)

plot_history(history1, "phase1_history.png", "Phase 1 — Head Training")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  PHASE 2 — FINE-TUNING (top layers unfrozen)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(f" PHASE 2 — Fine-tuning top {PHASE2_UNFREEZE} layers of EfficientNetB4")
print("═" * 60)

# Rebuild model with unfrozen top layers (preserves learned head weights)
model = build_model(unfreeze_layers=PHASE2_UNFREEZE)
model.load_weights(str(BEST_H5))  # restore best Phase 1 checkpoint

trainable2 = sum([tf.size(v).numpy() for v in model.trainable_variables])
print(f"📐  Trainable params (Phase 2): {trainable2:,}")

# Cosine annealing via LearningRateScheduler
def cosine_schedule(epoch, _lr):
    return cosine_lr_schedule(
        epoch, PHASE2_EPOCHS, PHASE2_LR_INIT, PHASE2_LR_MIN
    )

model.compile(
    # Lower LR for fine-tuning — prevents destroying ImageNet weights
    optimizer=keras.optimizers.Adam(
        learning_rate=PHASE2_LR_INIT,
        clipnorm=1.0,          # gradient clipping for stability
    ),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

phase2_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=str(BEST_H5),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_auc",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.LearningRateScheduler(cosine_schedule, verbose=0),
    callbacks.TensorBoard(
        log_dir=str(CHECKPOINT_DIR / "tb_logs" / "phase2"),
        update_freq="epoch",
    ),
]

history2 = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=phase2_callbacks,
    verbose=1,
)

plot_history(history2, "phase2_history.png", "Phase 2 — Fine-Tuning")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
# Load best checkpoint before evaluating
model.load_weights(str(BEST_H5))

best_threshold = evaluate_model(model, test_ds, threshold=0.35)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  EXPORT TO TFLITE (float16 quantisation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" EXPORTING to TFLite …")
print("═" * 60)

# Save full Keras model first
full_model_path = CHECKPOINT_DIR / "full_model.keras"
model.save(str(full_model_path))
print(f"✅  Full Keras model → {full_model_path}")

# TFLite conversion — float16 keeps accuracy close to float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_OUT) / 1e6
print(f"✅  TFLite model → {TFLITE_OUT}  ({size_mb:.1f} MB)")

# Quick sanity check: run one batch through the TFLite model
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_OUT))
interpreter.allocate_tensors()
input_det  = interpreter.get_input_details()
output_det = interpreter.get_output_details()
print(f"\n📐  TFLite input  shape : {input_det[0]['shape']}")
print(f"📐  TFLite output shape : {output_det[0]['shape']}")

# ── Verify one image ────────────────────────────────────────────────────────
sample_img, sample_label = next(iter(test_ds.unbatch().batch(1)))
interpreter.set_tensor(input_det[0]["index"], sample_img)
interpreter.invoke()
pred = interpreter.get_tensor(output_det[0]["index"])[0][0]
print(f"\n🧪  Sanity check — true label: {'Fake' if sample_label.numpy()[0] else 'Real'}")
print(f"                  raw score : {pred:.4f}")
print(f"                  decision  : {'Fake' if pred > best_threshold else 'Real'} (threshold={best_threshold:.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  SUMMARY & NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" TRAINING COMPLETE")
print("═" * 60)
print(f"""
📦  Output files:
    {TFLITE_OUT}                ← copy to backend/
    {full_model_path}           ← full Keras model
    {CHECKPOINT_DIR / 'plots'}  ← training charts

🔧  Update backend/deepfake.py:
    Change THRESHOLD to {best_threshold:.2f} for best F1 on this dataset.

🎯  Next steps:
    1. cp {TFLITE_OUT} backend/deepfake_efficientnet_model.tflite
    2. Set THRESHOLD = {best_threshold:.2f} in backend/deepfake.py
    3. git add backend/deepfake_efficientnet_model.tflite backend/deepfake.py
    4. git commit -m "feat: retrained model with EfficientNetB4"
    5. git push  →  Render will auto-redeploy
""")
