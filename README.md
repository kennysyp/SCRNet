**⚠ Note: This repository is under active peer-review management**

📜 **Associated Manuscript:**
- **Title:** SCRNet: A Self-Correcting Recurrent Network for Unsupervised Medical Image Registration
- **Status:** AE Assigns Reviewers at [IEEE TMI] | Preprint available on TechRxiv
- **Submitted Date:** 06-Dec-2025

🔒 **Important Notice for Reviewers and Users:**
1.  This is the **official and complete implementation** accompanying our submitted manuscript. The link to this repo has been provided to the journal's reviewing system.
2.  A preprint version of the paper is publicly available on TechRxiv. **Please refer to and cite the TechRxiv preprint** if you use this code for research during the review period.
3.  The code and models are shared to promote **transparency, reproducibility, and scientific collaboration**. We welcome feedback and academic discussions.
4.  **Usage:** We encourage the use of this code for **non-commercial research purposes**. For any other use, please contact the authors.

📈 **For Reviewers:** We appreciate your feedback. Any issues identified in the code are welcome and will be addressed in the camera-ready version upon acceptance.

---

Our **model weight file (.pth.tar)** is available at: <https://drive.google.com/drive/folders/1qU6qSA87-57nksJxgBq8-Iy-c4pgQNHm>

If you want to **run our model weight file directly**, please ensure your file directory is as follows:
```angular2html
SCRNet/
├── experiments/
│   ├── Abdomen_2025/
│   ├── IXI_2025/
│   ├── LPBA_2025/
│   ├── Mindboggle_2025/
│   └── OASIS_2025/
├── logs/
│   ├── Abdomen_2025/
│   ├── IXI_2025/
│   ├── LPBA_2025/
│   ├── Mindboggle_2025/
│   └── OASIS_2025/
├── mytools/
├── network/
├── infer.py
├── README.md
└── train.py
```

Create conda environment.
```shell
conda create -n your_env python=3.11
```

Activate conda enviroment.
```shell
conda activate your_env
```

When the author uploaded the code to the repository, there was no official version of pytorch for 5090 yet; only pre-release versions were available
Check your CUDA version. The author's CUDA is 12.8.
```shell
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Install the required packages.
```shell
python -m pip install tensorboard pystrum medpy nibabel scikit-image natsort einops tqdm faiss-cpu
```

Dataset:
- OASIS: <https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md>
- IXI: <https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md>
- LPBA: <https://github.com/tzayuan/TransMatch_TMI/blob/main/README.md>
- Mindboggle: <https://github.com/kharitz/learnpool/blob/master/README.md>
    - (Mindboggle101_individuals -> Mindboggle101_release3)
- Abdomen: <https://github.com/Runshi-Zhang/UTSRMorph/blob/main/README.md>


Train:
```shell
python train.py
```

Infer:
```shell
python infer.py
```