# NeuroVision AI – Guidance for the new brief (not the solution)

Scope of the brief: **3 models** (CNN from scratch, MobileNetV2 transfer learning, fine-tuning), then a comparison and error analysis.
Not in this brief: MLflow, a model registry, Streamlit/`predict_image`, Airflow, Vast.ai.

---

## 0. The big idea: one pipeline, three models, one comparison

All three models use the same data pipeline and the same split. Build one shared module and three notebooks:

```
data/raw/                 (gitignored)
src/data.py               scan, filter, load, resize, encode, split, tf.data builders
src/utils.py              plot_history(), evaluate_model(), plot_confusion(), save_results()
notebooks/
  01_data_eda.ipynb
  02_cnn_scratch.ipynb
  03_transfer_learning.ipynb
  04_fine_tuning.ipynb
  05_comparison.ipynb
models/                   *.keras (gitignored if big)
reports/                  results.json per model, figures
README.md  requirements.txt  .gitignore
```

**Why one shared split matters:** the comparison table is only fair if the CNN, the frozen model and the fine-tuned model are tested on exactly the same images. Fix the seed and save the split (indices or `.npz`) so every notebook reloads the same one.

**No MLflow here, so persist results yourself.** At the end of each training notebook, save a small `reports/<model>.json` with: test accuracy, macro F1, per-class recall, training time, `count_params()`, trainable params, and the history. Notebook 05 then reads the three JSONs and builds the table. This also lets you re-run a single notebook without redoing the others.

---

## 1. Data preparation (shared)

**Order of operations (the jury checks this):**
`scan -> filter extensions -> load + resize -> encode labels -> STRATIFIED split -> class_weight (train only) -> augment (train only) -> normalise`

1. **`cv2.imread` returns `None` on a corrupted file, it does not raise.** A try/except alone misses it. Check `None` too, and log how many files were skipped and why.
2. **Extensions:** lowercase before comparing (`.JPG` exists). Report what was rejected.
3. **BGR vs RGB:** OpenCV loads BGR. Convert to RGB before MobileNet. Make sure you always have 3 channels.
4. **Memory:** about 7k images at 224x224x3 in `float32` is about 4 GB. Keep arrays as `uint8` (about 1 GB) and cast inside the pipeline.
5. **Duplicates:** public brain-MRI sets can contain near-duplicates. If the same image is in train and test, that is leakage. A quick MD5 hash check before splitting is cheap, and a good answer to a code-review question.
6. **Split:** `train_test_split(..., stratify=y, random_state=SEED)` twice (train+val vs test, then train vs val). Typical: 70/15/15.
7. **Imbalance:** read the bar chart first. If the gap is small, say so and justify doing nothing. Otherwise use `class_weight` computed on train only.
8. **Labels:** `LabelEncoder` then `to_categorical` goes with `categorical_crossentropy`. (`sparse_categorical_crossentropy` also works without one-hot; just stay consistent.)

**Two normalisations, one per family of models (explicit in this brief):**
- CNN from scratch: `/255` gives [0, 1].
- MobileNetV2: `tf.keras.applications.mobilenet_v2.preprocess_input` gives **[-1, 1]**.
- Never apply both to the same model. That silently ruins the pretrained weights.
- Show one image before and after each, and print min/max, for the "compare before/after" requirement.
- Easiest clean design: keep the raw `uint8` arrays in the dataset, and put the normalisation step as the first step of each model (a `Rescaling` layer for the CNN, `preprocess_input` for MobileNet).

**Augmentation (train only):** `RandomRotation(~0.05)`, `RandomZoom(~0.1)`, `RandomTranslation(~0.1)`. Avoid flips unless you can justify them for MRI. Keras augmentation layers are inactive at inference when placed inside the model; if you use `tf.data.map`, apply it to the train dataset only.

**tf.data:** `.cache().shuffle(buffer).batch(32).prefetch(tf.data.AUTOTUNE)`. Cache before augmentation, otherwise the augmentation is frozen after epoch 1. Time the loop with and without it as proof.

---

## 2. Model 1: CNN from scratch

- 3-4 blocks `Conv2D -> (BatchNorm) -> MaxPool` with filters doubling, then `GlobalAveragePooling` or `Flatten`, `Dropout`, `Dense`, `Dense(4, softmax)`.
- ReLU in hidden layers; softmax + categorical cross-entropy because the 4 classes are mutually exclusive.
- Adam, lr 1e-3 to start. Think about image size (224 is required by the brief) and batch size (32), and be ready to justify them.
- Know the parameter formula: Conv2D params = (k*k*Cin + 1) * Cout.
- Callbacks: `ModelCheckpoint(save_best_only=True, monitor='val_loss')`, `EarlyStopping(restore_best_weights=True)`, `BackupAndRestore`. Wrap `time.time()` around `fit()`.
- Read the curves honestly: val loss rising while train loss falls is overfitting.

---

## 3. Model 2: transfer learning

- `MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))`, then `base.trainable = False`.
- Head: `GlobalAveragePooling2D -> Dropout(0.2-0.3) -> Dense(4, softmax)`.
- **Call the base with `training=False`:** `x = base(x, training=False)`. This keeps BatchNorm statistics frozen and is the classic bug if forgotten.
- `model.summary()`: only the head is trainable (about 5k params), about 2.2M are non-trainable. Be able to explain why.
- lr 1e-3, a few epochs, and the same callbacks (`ModelCheckpoint`, `EarlyStopping`).
- Curves: stagnating val while train keeps improving means overfitting.

---

## 4. Model 3: fine-tuning

- Restart from the best transfer-learning model.
- `base.trainable = True`, then re-freeze everything before a chosen index. MobileNetV2 has 154 layers; try the last 30-50. Keep BatchNorm layers frozen.
- **Recompile after changing `trainable`, or nothing changes.**
- lr about 1e-5 to 1e-4. Why: the weights are already good, and a large step erases what ImageNet taught.
- Re-check `model.summary()`: the trainable count must have grown. Time this phase separately.
- Reload the best checkpoint before the final evaluation.

---

## 5. Evaluation and test-set discipline

The brief asks for a test evaluation of each model. Two rules to state in the demo:
- **Never tune anything after looking at test results.** All choices (epochs, lr, layers to unfreeze, which checkpoint) are made on validation.
- Evaluate each model on test **once**, at the end of its notebook, with `confusion_matrix` and `classification_report` (use `target_names`).

**Tumour focus (especially model 3):** the worst error is a tumour predicted as `notumor` (a false negative). Report recall for glioma, meningioma and pituitary, and point to the cells in the matrix. A collapsed "tumour vs no tumour" view is a nice extra. Glioma and meningioma are the classic confusion pair.

---

## 6. Comparison and error analysis (notebook 05)

- Table with one row per model: accuracy, macro F1, training time, total params, trainable params. Load it from `reports/*.json`.
- Before vs after fine-tuning: gain in accuracy and, more importantly, in per-class recall.
- Grid of correct and incorrect predictions with true / predicted class and confidence; comment on which classes are confused and why.
- Conclude with a justified best model: metrics, tumour false negatives, and the cost (time, parameters). A bigger model is not automatically a better answer.
- Optional, for the "explainable AI" objective: Grad-CAM on a few images (last conv layer of MobileNetV2: `out_relu`).

---

## 7. Environment notes

- Native Windows TensorFlow (2.11 and later) has no GPU support. MobileNetV2 on about 7k images is OK on CPU but slow; for faster runs use WSL2, Colab or Kaggle. Whatever you use, put it in the README.
- Pin versions in `requirements.txt` (tensorflow, opencv-python-headless, scikit-learn, matplotlib, seaborn, pandas). Keras 3 saves `.keras`.
- `.gitignore`: `data/`, `models/*.keras`, `__pycache__/`, `.ipynb_checkpoints/`. Colab/Kaggle sessions are disposable, so download your models and reports.
- Small, regular commits are evidence of your work.

---

## 8. Jira

Use `neurovision_jira_v2.csv`: 6 epics, 20 stories, 48 subtasks, 67 points, one epic per block of the brief. Dates follow 12/10 to 16/10/2026.

---

## 9. Questions to be able to answer out loud

- Why this preprocessing order, and why is the split before augmentation?
- Why `/255` for the CNN and `preprocess_input` for MobileNet?
- What do trainable vs non-trainable parameters mean here?
- Why `training=False`, why recompile, why a much smaller learning rate?
- What is in the test set, and when did you first touch it?
- Which error is worst for the doctor, and which number shows it?
- Which model is best, and why not simply the biggest?
