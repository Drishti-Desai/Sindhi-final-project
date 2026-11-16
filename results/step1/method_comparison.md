# Step 1 method comparison (synthetic benchmark, known ground truth)

6 samples per page condition, 60 glyph samples, seed 7. Pages: method chosen by worst-condition F1; glyphs: by mean F1. Cells: text-pixel F1 / diacritic-dot recall. `oracle` = best possible single global threshold chosen with the ground truth (a ceiling, not a method).

| condition | otsu_raw | rembg_otsu | flatten_otsu | flatten_adaptive | flatten_sauvola | flatten_normalized_fixed | oracle |
|---|---|---|---|---|---|---|---|
| clean | 0.992 / 1.00 | 0.079 / 0.50 | 0.992 / 1.00 | 0.964 / 1.00 | 0.961 / 1.00 | 0.972 / 1.00 | 0.992 |
| transparent | 0.992 / 1.00 | 0.068 / 0.50 | 0.992 / 1.00 | 0.966 / 1.00 | 0.959 / 1.00 | 0.969 / 1.00 | 0.992 |
| colored | 0.992 / 1.00 | 0.057 / 0.36 | 0.992 / 1.00 | 0.973 / 1.00 | 0.976 / 1.00 | 0.972 / 1.00 | 0.993 |
| gradient | 0.232 / 1.00 | 0.051 / 0.30 | 0.992 / 1.00 | 0.970 / 1.00 | 0.968 / 1.00 | 0.973 / 1.00 | 0.992 |
| shadow | 0.214 / 1.00 | 0.065 / 0.26 | 0.993 / 1.00 | 0.969 / 1.00 | 0.966 / 1.00 | 0.975 / 1.00 | 0.994 |
| texture | 0.987 / 1.00 | 0.272 / 0.37 | 0.987 / 1.00 | 0.671 / 1.00 | 0.961 / 1.00 | 0.967 / 1.00 | 0.987 |
| low_contrast | 0.992 / 1.00 | 0.094 / 0.29 | 0.992 / 1.00 | 0.978 / 1.00 | 0.931 / 0.93 | 0.969 / 1.00 | 0.993 |
| dark | 0.991 / 1.00 | 0.039 / 0.24 | 0.991 / 1.00 | 0.958 / 1.00 | 0.958 / 1.00 | 0.967 / 1.00 | 0.992 |
| noise | 0.887 / 1.00 | 0.468 / 0.56 | 0.967 / 1.00 | 0.902 / 1.00 | 0.950 / 1.00 | 0.937 / 1.00 | 0.889 |
| ruled | 0.637 / 1.00 | 0.377 / 0.64 | 0.955 / 1.00 | 0.944 / 1.00 | 0.952 / 1.00 | 0.931 / 1.00 | 0.941 |
| pencil_faint | 0.937 / 0.99 | 0.172 / 0.29 | 0.822 / 1.00 | 0.453 / 1.00 | 0.937 / 0.98 | 0.899 / 0.88 | 0.945 |
| desk | 0.105 / 1.00 | 0.847 / 1.00 | 0.941 / 1.00 | 0.954 / 1.00 | 0.953 / 1.00 | 0.958 / 1.00 | 0.896 |
| glyph_small | 0.119 / 0.97 | 0.196 / 0.13 | 0.517 / 0.99 | 0.680 / 0.80 | 0.281 / 0.21 | 0.738 / 0.94 | 0.752 |

| summary | otsu_raw | rembg_otsu | flatten_otsu | flatten_adaptive | flatten_sauvola | flatten_normalized_fixed |
|---|---|---|---|---|---|---|
| page_mean_f1 | 0.747 | 0.216 | 0.968 | 0.892 | 0.956 | 0.957 |
| page_worst_f1 | 0.105 | 0.039 | 0.822 | 0.453 | 0.931 | 0.899 |
| page_dot_recall | 0.999 | 0.444 | 1.000 | 1.000 | 0.993 | 0.990 |
| glyph_f1 | 0.119 | 0.196 | 0.517 | 0.680 | 0.281 | 0.738 |
| glyph_p10_f1 | 0.052 | 0.000 | 0.085 | 0.279 | 0.000 | 0.532 |
| glyph_dot_recall | 0.969 | 0.133 | 0.994 | 0.797 | 0.211 | 0.944 |
| seconds_per_image | 0.006 | 1.801 | 0.192 | 0.204 | 0.210 | 0.203 |

Chosen for pages: **flatten_sauvola**; for letter crops: **flatten_normalized_fixed**.
Deskew: mean |error| 0.03 deg, max 0.09 deg.
Condition detection rate (expected step fired): dark: 100%, noise: 100%, low_contrast: 100%, pencil_faint: 100%
