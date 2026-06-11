

---

# 📖 README.md

```markdown
# Libyan Currency Recognition for the Visually Impaired 🇱🇾👁️

An end-to-end Assistive Technology application that leverages Artificial Intelligence to recognize Libyan currency denominations. Designed specifically with robust **Web Accessibility (A11y)** features, the application provides an instant audio-tactile response to assist blind and visually impaired individuals in handling daily financial transactions independently.

---

## 🌟 Key Features

### 🧠 Deep Learning Architecture & AI
* **Transfer Learning Engine:** Built on top of **MobileNetV2** (pre-trained on ImageNet) for rapid, resource-efficient edge-compatible classification.
* **Dual-Stage Augmentation Pipeline:** Combines offline preprocessing scripts with live real-time training generators to simulate unpredictable real-world captures.
* **Explainable AI (XAI):** Features a visual heatmap integration (Grad-CAM ready) allowing caretakers or developers to see exactly *where* the AI focused its attention to make a decision.

### ♿ Accessibility-First Design (A11y)
* **Omnipresent Camera Trigger:** Clicking anywhere on the screen immediately triggers the camera interface or file picker, eliminating the need to locate fine UI buttons visually.
* **Screen Reader Optimization:** Integrated live regions via `aria-live="assertive"` for dynamic screen announcements at critical phases (e.g., uploading, analyzing, and final results).
* **Scalable Typography:** Real-time font scaling capabilities through dedicated high-contrast controls to assist users with low-vision or partial sight.
* **Full Keyboard Navigation:** Optimized Focus Management including standard focus rings, `tabindex` flows, and a custom skip-link element (`.skip-link`).

---

## 🗂️ Project Structure

```text
├── dataset_original/       # Raw collected currency images categorized by class
├── dataset_augmented/      # Generated variations from custom preprocessing script
├── augment.py              # Custom PIL/NumPy offline data augmentation script
├── train.py                # TensorFlow/Keras training notebook (Google Colab optimized)
├── index.html              # Accessibility-first frontend web UI (RTL, Audio-integrated)
├── requirements.txt        # Backend dependencies and framework versions
└── README.md               # Repository documentation

```

---

## 📊 Dataset & Augmentation Pipeline

### 🔗 Dataset Access

The base image repository containing both initial collections and augmented variations can be accessed here:

> 📂 **Dataset Link:** [first_dataset - Google Drive](https://www.google.com/search?q=https://drive.google.com/drive/folders/170gL6wV_2i2oD9P1vWl-F8f9-e_R2jM9%3Fusp%3Dsharing) *(Replace with your actual public shareable URL link if necessary)*

### 🔄 Preprocessing Strategy (`augment.py`)

Because the application relies heavily on unstable camera pictures taken by visually impaired users, images are explicitly synthetically expanded by **20x (`SAMPLES_PER_IMAGE = 20`)** per original capture using structural and environment variances:

1. **360° Free Rotation:** Rotates images across a full random spectrum (`0-359` degrees).
2. **Smart Zooming:** Applies automated random cropping (`1.1x` to `1.4x`) and programmatic canvas boxing (`0.7x` to `0.9x`).
3. **Luminance Stressing:** Modulates brightness levels between `0.6` (dim low-light) and `1.5` (over-exposed sunlight).
4. **Uniform Scaling:** Pipelines everything down to a structured shape of `224x224` pixels.

---

## 🚀 Model Training Details (`train.py`)

The neural network is trained using **TensorFlow/Keras** with transfer learning optimizations.

* **Base Network:** MobileNetV2 (Frozen ImageNet weights).
* **Classifier Head:** Global Average Pooling 2D ➡️ Dropout Layer (30% rate) ➡️ Softmax Output Dense Layer (10 Class Categorization).
* **Loss & Optimizer:** `Adam` Optimizer ($\text{Learning Rate} = 0.0001$) paired with `Categorical Crossentropy`.
* **Live Data Generator:** Applies additional multi-directional adjustments (shear, shift, horizontal flips) during runtime training epochs.

```python
# To train the model, run the script within your Google Colab workspace:
python train.py

```

*Outputs a production-ready weights archive named `my_currency_model.h5` upon completion.*

---

## 🌐 Web Client Execution

The UI is built purely with accessibility mechanics, integrating direct hardware captures (`capture="environment"`), custom CSS scaling variables, and reactive sound feedback triggers.

### Dependencies Installation

Install required system and backend framework packages via pip:

```bash
pip install -r requirements.txt

```

### Quick Deployment (Local Testing)

You can deploy a mock server or host the system via Flask to link the `index.html` interface directly with your compiled `.h5` model:

```bash
# Example if utilizing Gunicorn in production environments
gunicorn app:app

```

---

## 🛠️ Technology Stack

* **Machine Learning Core:** TensorFlow, NumPy, OpenCV, Pillow.
* **Frontend Interfaces:** HTML5 (Semantic Structure), CSS3 (Flexbox/Grid/Variables), Vanilla JavaScript (A11y Event Observers).
* **A11y Toolkit:** ARIA Live Regions, Screen-reader Assistive Utilities, Font-Awesome Icon kits.

```

```
