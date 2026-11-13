# Vision–language detection of drug-induced QT prolongation at risk of Torsades de Pointes

Code for:

> D. Labrador Alonso and A. Alahmadi. *Vision–language detection of drug-induced QT prolongation at risk of Torsades de Pointes from pseudo-coloured ECG images: do T-wave morphology and colour context help or hinder?* Biomedical Signal Processing and Control (under revision, BSPC-D-26-17774).

The study asks whether a biomedical vision–language model (BiomedCLIP) can tell ECG beats with QT prolongation at risk of Torsades de Pointes (TdP) from low-risk beats, and whether a QT-nomogram colour overlay and text prompts that name colour or T-wave shape help.

## Data

ECGs come from the **ECGRDVQ** study on PhysioNet ("ECG effects of ranolazine, dofetilide, verapamil, and quinidine in healthy subjects"): 22 healthy volunteers, five treatment periods in a crossover design, triplicate 10-s 12-lead ECGs at 16 time points. The data are not redistributed here; download them from PhysioNet under its licence.

Each recording is labelled **at risk** when its QT lies on or above the QT-nomogram line for its heart rate (QT–nomogram distance ≥ 0 ms), and **low risk** otherwise.

## Repository layout

| Path | Contents |
|---|---|
| `rendering/` | How each ECG is turned into a beat image (pseudo-coloured and colourless) |
| `notebooks/01_revision_analysis.ipynb` | Full analysis for the revised paper: data checks, embeddings, zero-shot inference, trained probes, cross-validation schemes, nested selection, QT–heart-rate baseline, bootstrap comparisons, rendering-geometry comparison and reproduction of the originally published numbers |
| `notebooks/02_supplementary_material.ipynb` | Builds the Supplementary Material tables and the Fig. 2 example panels from the analysis outputs |
| `src/make_supplement.py` | Code used by notebook 02 |
| `original_submission/` | Notebooks used for the originally submitted version (zero-shot and trained probes, coloured and colourless) |
| `results/` | Analysis outputs (CSV) and the software environment used |

## How to reproduce

1. Download ECGRDVQ from PhysioNet and render the images (see `rendering/`). You need `Colour_Heartbeat.zip`, `Colourless_Heartbeat.zip` and a metadata table with one row per recording (recording number, participant, treatment, period, time point, QT, RR, heart rate, QT–nomogram distance).
2. Open `notebooks/01_revision_analysis.ipynb` in Google Colab with a GPU, set the paths in the **Config** cell and run all cells (about 30 minutes on a T4). Set `SMOKE_TEST = True` first to check the set-up on synthetic data.
3. Run `notebooks/02_supplementary_material.ipynb` (CPU is enough) to build the supplementary tables and figure panels.

## Methods in brief

- **Model:** BiomedCLIP (`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) through `open_clip`, frozen.
- **Prompts:** four context strategies (colour and/or morphology named or not) crossed with four label sets (no label, L1–L3).
- **Probes:** logistic regression on standardised image embeddings, image + prompt features, or prompt features alone (C = 1, balanced class weights, seed 42).
- **Evaluation:** five-fold participant-grouped cross-validation (main analysis), with image-level and triplicate-grouped splits for comparison; thresholds tuned by inner cross-validation; participant-cluster bootstrap (2,000 resamples) with Holm adjustment for paired comparisons.
- **Baseline:** logistic regression on QT and heart rate.

Exact package versions, the model revision and the hardware are recorded in `results/environment.json`.

## Licence

MIT (see `LICENSE`). The ECGRDVQ data keep their own PhysioNet licence.

## Citation

Please cite the article above once published.
