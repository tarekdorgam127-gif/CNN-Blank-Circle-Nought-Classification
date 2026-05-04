#  CNN Image Classification — Blank / Circle / Nought

##   المشروع بيعمل إيه؟

المشروع بيدرّب شبكة عصبية تلافيفية (CNN) تصنّف الصور إلى 3 فئات:

| الفئة | الوصف |
|-------|--------|
| `blank` | صورة فارغة بدون رسم |
| `circle` |  دائرة |
| `nought` |   إكس |

الهدف هو بناء موديل يقدر يفرق بين الصور دي تلقائيًا باستخدام الـ Deep Learning.

---

##  إزاي نشغّله؟

### 1. جهّز البيئة
تأكد إن عندك Python و PyTorch مثبتين:

```bash
pip install torch torchvision
```

### 2. جهّز البيانات
- فك ضغط ملفات الـ ZIP:
  - `Blanck-Circle-Nought-dataset-trian.zip` → مجلد `data/train`
  - `Blanck-Circle-Nought-datase-test.zip` → مجلد `data/test`
- تأكد إن كل فئة في مجلدها الخاص داخل `train/` و `test/`

### 3. شغّل الـ Notebook
افتح الملف ده وشغّل الـ cells بالترتيب:

```
CNN_classification_blanck_circle_nought.ipynb
```

> **ملاحظة:** الكود الأصلي اتشغّل على **Google Colab** مع GPU من نوع **T4**.  
> لو بتشغّله locally، غيّر مسار البيانات من `/content/drive/MyDrive/data/` للمسار عندك.

---

##  اللي استخدمته

### المكتبات

| المكتبة | الاستخدام |
|---------|-----------|
| `torch` | بناء وتدريب الموديل |
| `torch.nn` | طبقات الشبكة العصبية |
| `torchvision.transforms` | معالجة الصور (Resize, Normalize) |
| `torchvision.datasets.ImageFolder` | تحميل البيانات من المجلدات |
| `torch.utils.data.DataLoader` | تحميل الـ batches |

### الموديل — `CustomCNN`

شبكة CNN مبنية from scratch بتتكون من:

```
Input (3 × 28 × 28)
    ↓
Conv2d(3 → 16, kernel=3) + ReLU + MaxPool → (16 × 14 × 14)
    ↓
Conv2d(16 → 32, kernel=3) + ReLU + MaxPool → (32 × 7 × 7)
    ↓
Flatten → 1568
    ↓
Linear(1568 → 128) + ReLU + Dropout(0.6)
    ↓
Linear(128 → 3)  ← Output (3 classes)
```

### إعدادات التدريب

| الإعداد | القيمة |
|---------|--------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Regularization | Weight Decay = 0.0005 (L2) |
| Batch Size | 10 |
| Epochs | 8 |

### نتائج التدريب

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.7940 | 58.1% |
| 4 | 0.2473 | 90.4% |
| 8 | 0.0289 | **99.2%** |

**نتيجة الـ Test:** `Loss: 0.0049 — Accuracy: 100%` 
