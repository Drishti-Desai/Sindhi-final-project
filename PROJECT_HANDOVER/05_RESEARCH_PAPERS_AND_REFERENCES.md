# 05 — Research papers, sources and references

Three groups are kept apart:

- **A. Found in the repository**: everything the project files themselves cite or point to (audit of all tracked
  files, READMEs and notebooks; details with file:line in `appendix/C_raw_models_references_safety.md`, Part B).
- **B. Original papers of the methods and tools we use**: the code uses these methods by name but **does not cite
  them**. Standard, widely cited references are listed so the report can cite them; check each against the paper
  before final submission.
- **C. Sindhi OCR literature**: found by a web search on 2026-11-16 (not in the repository). Only details shown in the
  search results are given; missing details are marked NOT VERIFIED.

---

## A. References found in the repository

| # | Resource | Authors / owner | Year | Identifier | What we used | Implemented / adapted / studied |
|---|---|---|---|---|---|---|
| A1 | "POS Tagging on Handwritten Sindhi Sentences", Mendeley Data, V1 | Maria Soomro, Muhammad Khalid (University of Sindh) | 2024 | doi:10.17632/phk66sgmp5.1; https://data.mendeley.com/datasets/phk66sgmp5/1 | 998 handwritten sentence images, inspected; **not used for training** (no transcriptions) | studied |
| A2 | Wikipedia (Sindhi), dataset `wikimedia/wikipedia`, config `20231101.sd` | Wikimedia Foundation / Wikipedia contributors | 2023 dump | https://huggingface.co/datasets/wikimedia/wikipedia | Sindhi text for word lists, lexicon, masked-LM training | used (data) |
| A3 | Roboflow Universe "sindhi-ocrf" | workspace `iitbresearchwork` | export 2023-07-17 | https://universe.roboflow.com/iitbresearchwork/sindhi-ocrf | printed pages used only to check preprocessing | used (data, evaluation only) |
| A4 | "Sindhi POS Tagging Dataset and Model" (`Sindhi-Pos-Tagging-main`) | **NOT FOUND** (README has a placeholder URL) | NOT FOUND | NOT FOUND | 676 POS-tagged sentences → 518 clean sentences for the lexicon and LM | used (data); CRF model and notebooks **not used** |
| A5 | `Sindhi_Legal_Dataset.csv` | **NOT FOUND** | NOT FOUND | NOT FOUND | legal Q&A text for training text and as the source of handwritten page texts | used (data) |
| A6 | xlm-roberta-base (model weights) | Facebook AI | — | Hub id not recorded in the repository; local cache folder `models--FacebookAI--xlm-roberta-base` | base model for correction | adapted (further trained) |
| A7 | Tesseract 5.5.2 (`snd`, `urd`, `ara`, `script/Arabic`), `text2image` | Tesseract OCR project | — | system install | Step 3 evaluation; printed word rendering | used (tool) |
| A8 | rembg (U²-Net, onnxruntime) | `rembg` package | — | `requirements.txt` (rembg 2.0.85) | Step 1 comparison, rejected | studied |
| A9 | "TrOCR-style, segmentation-free" design note | — | — | `PLAN.md` | inspiration for the word recognizer (CNN encoder + Transformer decoder) | adapted (idea) |
| A10 | Candidate corpora **never downloaded**: `HuggingFaceFW/fineweb-2` (snd_Arab), CC-100 `sd.txt.xz`, `DanishMahdi/snd-headline-ocr` | — | — | `PLAN.md` | not used | — |

**No research paper about a method is cited anywhere in the repository**; the only DOI is the Mendeley dataset (A1).
No external code was copied into the project (the third-party POS notebooks sit unused in `data/raw/pos_text/`).

## B. Original references for the methods and tools used (for the report)

| Method / tool used | Where in our code | Reference | Use |
|---|---|---|---|
| Sauvola binarisation | `preprocess/ops.py::binarize` | J. Sauvola and M. Pietikäinen, 2000 | implemented via scikit-image |
| Otsu threshold | `preprocess/ops.py` | N. Otsu, 1979 | used (comparison, helpers) |
| CLAHE | `preprocess/ops.py::apply_clahe` | K. Zuiderveld, 1994 (Graphics Gems IV) | used via OpenCV |
| Non-local means denoising | `preprocess/ops.py::denoise` | A. Buades, B. Coll, J.-M. Morel, 2005 | used via OpenCV |
| Residual CNN | `cnn_transformer/models.py::BasicBlock`, `Trunk` | K. He *et al.*, 2016 | adapted (small ResNet) |
| Transformer encoder/decoder | `WordRecognizer` | A. Vaswani *et al.*, 2017 | implemented via PyTorch |
| Masked language modelling, whole-word masking | `scripts/step4_train_mlm.py` | J. Devlin *et al.*, 2019 (BERT) | adapted |
| XLM-RoBERTa | Step 4 | A. Conneau *et al.*, 2020 | adapted (further training, token head) |
| Label smoothing | Step 2 losses | C. Szegedy *et al.*, 2016 | used |
| AdamW | all training | I. Loshchilov and F. Hutter, 2019 | used |
| Gradient checkpointing | `WordRecognizer.run_trunk`, HF Trainer | T. Chen *et al.*, 2016 | used |
| Levenshtein distance (CER/WER) | `common/metrics.py` | V. I. Levenshtein, 1966 | implemented |
| Hungarian assignment | `tesseract_ft/boxes.py::score_boxes` | H. W. Kuhn, 1955 | used via SciPy |
| Symmetric-delete spelling lookup | `xlmr/correct.py::deletes`, `Lexicon.near` | W. Garbe, SymSpell (GitHub, software) | idea adapted (our own code) |
| U²-Net (rembg) | `preprocess/segmentation.py` | X. Qin *et al.*, 2020 | studied, rejected |
| Tesseract | Step 3 | R. Smith, 2007 | evaluated |
| TrOCR | design idea / future work | M. Li *et al.*, 2023 (AAAI) | studied only |
| CRNN + CTC | future work | B. Shi, X. Bai, C. Yao, 2017; A. Graves *et al.*, 2006 | studied only |
| PyTorch | everything | A. Paszke *et al.*, 2019 | used |
| Hugging Face Transformers | Step 4 | T. Wolf *et al.*, 2020 | used |
| OpenCV | image processing | G. Bradski, 2000 | used |
| scikit-image | Sauvola | S. van der Walt *et al.*, 2014 | used |
| Gradio | `app.py` | A. Abid *et al.*, 2019 (arXiv:1906.02569) | used |

## C. Sindhi / Urdu-like script OCR literature (web search, 2026-11-16)

See `06_PROJECT_UNIQUENESS_AND_INNOVATION.md` §3 for what each work covers. These were **studied only** (titles and
abstracts via search results); none of their code or data was used.

---

## IEEE-style reference list

Entries marked † were taken from web search results and still need checking against the publication (authors,
volume, pages). Entries in group B are standard references; verify page numbers before submission.

[1] M. Soomro and M. Khalid, "POS Tagging on Handwritten Sindhi Sentences," Mendeley Data, V1, 2024, doi: 10.17632/phk66sgmp5.1.
[2] Wikimedia Foundation, "Wikipedia dataset (wikimedia/wikipedia), Sindhi 20231101.sd," Hugging Face. [Online]. Available: https://huggingface.co/datasets/wikimedia/wikipedia
[3] Roboflow user "iitbresearchwork," "Sindhi OCRf dataset," Roboflow Universe, 2023. [Online]. Available: https://universe.roboflow.com/iitbresearchwork/sindhi-ocrf
[4] J. Sauvola and M. Pietikäinen, "Adaptive document image binarization," *Pattern Recognition*, vol. 33, no. 2, pp. 225–236, 2000.
[5] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, 1979.
[6] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, P. S. Heckbert, Ed. Academic Press, 1994, pp. 474–485.
[7] A. Buades, B. Coll, and J.-M. Morel, "A non-local algorithm for image denoising," in *Proc. IEEE CVPR*, 2005, pp. 60–65.
[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.
[9] A. Vaswani *et al.*, "Attention is all you need," in *Proc. NeurIPS*, 2017, pp. 5998–6008.
[10] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
[11] A. Conneau *et al.*, "Unsupervised cross-lingual representation learning at scale," in *Proc. ACL*, 2020, pp. 8440–8451.
[12] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in *Proc. IEEE CVPR*, 2016, pp. 2818–2826.
[13] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.
[14] T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training deep nets with sublinear memory cost," arXiv:1604.06174, 2016.
[15] V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," *Soviet Physics Doklady*, vol. 10, no. 8, pp. 707–710, 1966.
[16] H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Research Logistics Quarterly*, vol. 2, no. 1–2, pp. 83–97, 1955.
[17] W. Garbe, "SymSpell: symmetric delete spelling correction algorithm," GitHub repository. [Online]. Available: https://github.com/wolfgarbe/SymSpell
[18] X. Qin, Z. Zhang, C. Huang, M. Dehghan, O. R. Zaiane, and M. Jagersand, "U²-Net: Going deeper with nested U-structure for salient object detection," *Pattern Recognition*, vol. 106, 107404, 2020.
[19] R. Smith, "An overview of the Tesseract OCR engine," in *Proc. ICDAR*, 2007, pp. 629–633.
[20] M. Li *et al.*, "TrOCR: Transformer-based optical character recognition with pre-trained models," in *Proc. AAAI*, 2023.
[21] B. Shi, X. Bai, and C. Yao, "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition," *IEEE Trans. PAMI*, vol. 39, no. 11, pp. 2298–2304, 2017.
[22] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist temporal classification," in *Proc. ICML*, 2006, pp. 369–376.
[23] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019.
[24] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.
[25] G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, 2000.
[26] S. van der Walt *et al.*, "scikit-image: image processing in Python," *PeerJ*, vol. 2, e453, 2014.
[27] A. Abid, A. Abdalla, A. Abid, D. Khan, A. Alfozan, and J. Zou, "Gradio: Hassle-free sharing and testing of ML models in the wild," arXiv:1906.02569, 2019.
[28] † A. A. Sanjrani, J. Baber, M. Bakhtyar *et al.*, "Extended framework for Sindhi numerals OCR using gradient orientation histograms," *J. Intelligent & Fuzzy Systems*, 2022, doi: 10.3233/JIFS-219304.
[29] † "Handwritten optical character recognition system for Sindhi numerals," in *IEEE conf.*, 2016, IEEE Xplore doc. 7495235 (authors NOT VERIFIED).
[30] † "Handwritten Sindhi character recognition using neural networks," *Mehran Univ. Research J. Eng. Technol.*, 2018 (authors NOT VERIFIED).
[31] † A. A. Chandio *et al.*, "Deep learning-based isolated handwritten Sindhi character recognition," *Indian J. Science and Technology*, 2020 (co-authors NOT VERIFIED).
[32] † S. Naz *et al.*, "The optical character recognition of Urdu-like cursive scripts," *Pattern Recognition*, 2014.
[33] † Hakro *et al.* (initials NOT VERIFIED), "A study of Sindhi related and Arabic script adapted languages recognition," arXiv:1412.4217, 2014.
[34] † Awan, Hakro *et al.*, "A complete off-line Sindhi handwritten text recognition: a survey," 2017, Zenodo 3469359 (initials NOT VERIFIED).
[35] † Solangi *et al.*, "Offline printed Sindhi optical text recognition: survey," in *Proc. ICETAS*, 2018/2019 (details NOT VERIFIED).
[36] † "OHSCR: Benchmarks dataset for offline handwritten Sindhi character recognition," ~2024 (authors/venue NOT VERIFIED).

## References for the BTech report (recommended subset)

Core: [4] Sauvola, [8] ResNet, [9] Transformer, [10] BERT, [11] XLM-R, [15] Levenshtein, [19] Tesseract, [20] TrOCR,
[21] CRNN, [22] CTC, [23] PyTorch, [24] Transformers, [27] Gradio.
Sindhi context (after checking each paper): [28], [31], [32], [33], [34], [35]. Data: [1], [2] (and [3] if you
mention the preprocessing check). Add the legal and POS text sources once their origin is found.
