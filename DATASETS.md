# Datasets and models used — references and data protection

This project used the datasets below. **The raw datasets and trained model weights are not included in this
repository.** The result files in `results/` include evaluation outputs (metrics, recognized and reference text of
test words and pages) and 19 figures that show **handwriting excerpts** (letter crops, word crops, page examples) to
document the results. These excerpts remain the writers' property and must not be reused (see `LICENSE`). Writers
are identified only by letter codes (A, D, H, J, …); their names are not recorded anywhere in the code or
documentation.

## Personal / project-collected data (private — not published)

| Data | Description | Status |
|---|---|---|
| Handwritten Sindhi pages (first set) | 22 page photos with transcriptions, writers A, D, H, J | private; **not included**; may not be redistributed without the writers' written consent |
| Handwritten Sindhi pages (second batch) | 129 page photos, word crops checked by the team | private; **not included** |
| Isolated Sindhi letter images | 4,241 unique letter crops, 52 classes, from the team's earlier project (source and licence not recorded) | private; **not included** |
| Reviewed words saved from the app (`user_feedback/`) | optional user corrections | never published (git-ignored) |

## Third-party datasets

| # | Dataset | Source | Licence | How it was used | Included here? |
|---|---|---|---|---|---|
| [D1] | Wikipedia (Sindhi), `wikimedia/wikipedia`, config `20231101.sd` | Wikimedia Foundation / Wikipedia contributors — https://huggingface.co/datasets/wikimedia/wikipedia | CC BY-SA 3.0 + GFDL | Sindhi text for word lists, the lexicon and language-model training | no |
| [D2] | M. Soomro and M. Khalid, "POS Tagging on Handwritten Sindhi Sentences," Mendeley Data, V1, 2024, doi:10.17632/phk66sgmp5.1 (University of Sindh) | https://data.mendeley.com/datasets/phk66sgmp5/1 | CC BY 4.0 | inspected only; **not used for training** (no transcriptions) | no |
| [D3] | "Sindhi OCRf" dataset, Roboflow Universe, workspace `iitbresearchwork` (export 2023-07-17) | https://universe.roboflow.com/iitbresearchwork/sindhi-ocrf | CC BY 4.0 | printed pages used only to check image preprocessing | no |
| [D4] | Sindhi POS-tagged sentences (`Sindhi-Pos-Tagging`) | source repository and authors not recorded | **unknown** | 518 cleaned sentences used as training text | no |
| [D5] | Sindhi legal question–answer dataset (`Sindhi_Legal_Dataset.csv`) | source and authors not recorded | **unknown** | training text; source of the texts the writers copied | no |

## Pretrained models and tools

| # | Item | Source | Licence | Use |
|---|---|---|---|---|
| [M1] | xlm-roberta-base | Facebook AI (A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," ACL 2020) | MIT (per its model card) | further trained on Sindhi text for spelling correction |
| [M2] | Tesseract OCR 5.5.2 | Tesseract OCR project (R. Smith, ICDAR 2007) | Apache-2.0 | evaluated as a box finder; `text2image` rendered printed training words |
| [M3] | rembg / U²-Net | rembg package (X. Qin et al., Pattern Recognition 2020) | see package | evaluated and rejected |

Libraries and their versions are listed in `requirements.txt`. Full reference list:
`PROJECT_HANDOVER/05_RESEARCH_PAPERS_AND_REFERENCES.md`.

## Data protection notes

- The handwriting datasets are excluded; only the 19 result figures in `results/` show handwriting excerpts, and
  the result CSV/JSON files contain recognized and reference text of test words and pages.
- Trained model weights are not published, because they were trained on private handwriting and on text whose
  licence is unknown.
- Requests for data access must be made to the project team; access requires the writers' written consent.
