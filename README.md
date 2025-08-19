# YOLO  — Nesne Tespiti Çalışmaları

Bu depo, YOLO tabanlı nesne tespiti denemelerimi, eğitim/inferans notlarımı ve raporumu içerir.  
Odak: **hızlı prototipleme**, **transfer learning** ve **temel değerlendirme metrikleri**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nslzsn/yolo/blob/main/YOLOv12.ipynb)

---

## İçerik

- **01_training.ipynb** — Colab üzerinde eğitim/inferans denemeleri (örnek komutlar).
- **hedef.py** — Basit script (deneme amaçlı). (İleride `train.py` / `infer.py` olarak ayrıştırılabilir.)
- **best.pt** — Eğitilen ağırlık dosyası (*büyük dosyalar için ileride Releases veya Drive linki tercih edilebilir*).
- **rapor.pdf** — Kısa rapor / notlar.
- **README.md** — Bu dosya.



---

## Kurulum

Python 3.10+ önerilir.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
