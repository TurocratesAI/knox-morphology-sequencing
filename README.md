# KNOX

Grading-free prediction of biochemical recurrence (BCR) after radical prostatectomy from a single H&E whole-slide
image, built for the OWL challenge.

- **Encoders.** Three frozen pathology foundation models: Midnight-12K, UNI2-h and Phikon-v2.
- **Model.** For each encoder, 5 fold models: gated-attention multiple-instance learning (MIL) trained with a Cox
  partial-likelihood loss on RUMC and TCGA together. Each fold model's risk is z-scored; each encoder's score is the
  mean of its 5 fold models, z-scored again; the final score is the equal-weight mean over the three encoders.
- **Tiles.** 224 px at 0.5 um/px (112 um fields) inside the organiser's tissue mask, taken on a fixed lattice set by
  the slide extent alone and filtered for tissue. Whether a field is read depends only on its own position, so editing
  the mask can only remove reads; it never redraws the sample.
- **Maps.** Sequential knockout. The field whose removal lowers the score most is removed first, every effect is
  recomputed exactly on what is left (attention renormalises), and the process repeats. Each region's value is the
  exact change in the score if it were removed.
- **No grading.** No Gleason pattern, ISUP grade group, grade or stage is used anywhere in this code: not as input,
  target or selection criterion. The only labels are time to BCR and event status.

## Results

Out-of-fold Harrell's C of the three-encoder ensemble, computed within each cohort (RUMC / TCGA): 0.728 / 0.731 on the
folds used by the pre-set encoder selection rule (`training/e28_ensemble.py`); 0.710 / 0.723 on fresh folds, against
0.705 / 0.706 for Midnight-12K alone.

## Layout

```
container/   challenge container: inference.py (full pipeline), app.py (HTTP server), Dockerfile, build/test/save scripts
training/    common.py (paths, loaders), MIL runners, e28_ensemble.py (encoder selection), train_final_v2.py (final models)
features/    owl_extract.py (tiling, Midnight and H-optimus-0 features), e26_multi_encoder.py (UNI2-h, Phikon-v2),
             sparse_fetch.py, e0_tissue_filter.py, e2_dictionary.py, e7_nuclei.py, e9_cell_features.py
analysis/    deletion curves, compartment knockouts, readable surrogate, encoder agreement, mask-edit checks,
             sequential knockout (e12 to e33)
```

## Paths

All scripts take paths from one root, `KNOX_ROOT` (default: the current directory):

- `$KNOX_ROOT/data/`: `labels/training_labels.csv` (columns `case_id`, `event`, `follow_up_years`),
  `labels/tcga_bcr.csv` (`case_id`, `event`, `follow_up_y`), the slide list `jobs_all.tsv`, and everything the scripts
  write (`tiles/`, `feats/<encoder>/`, `tissue/`, `dict/`, `nuclei/`, `cellfeat/`, `exp/`).
- `$KNOX_ROOT/logs/exp/`: result JSONs. Create it before running.
- `$KNOX_MODEL_DIR` (default `$KNOX_ROOT/model`): trained MIL weights, `owl_v2/` (and `owl/` for the v1 analyses).

`jobs_all.tsv` is tab-separated, one slide per line: cohort (`owl` or `tcga`), case id, WSI source (an `s3://` URI
for the OWL training slides, or a path in the Hugging Face dataset `waticlems/tcga-prad-bcr` for TCGA) and tissue mask
source. TCGA downloads read `HF_TOKEN` from the environment.

## Encoder weights

Encoder weights are not included. Obtain them from their official sources and use them under their own licences:
Midnight-12K (`kaiko-ai/midnight`), UNI2-h (`MahmoodLab/UNI2-h`, gated), Phikon-v2 (`owkin/phikon-v2`) and, for feature
extraction, the tissue filter and the phenotype dictionary only, H-optimus-0 (`bioptimus/H-optimus-0`, gated).
`features/owl_extract.py` loads Midnight-12K and H-optimus-0 from the local Hugging Face cache (offline). Trained KNOX
weights are not included either.

## Training

```
export KNOX_ROOT=/path/to/workdir
python features/owl_extract.py --jobs $KNOX_ROOT/data/jobs_all.tsv --shard 0/1 --device cuda:0
python features/e0_tissue_filter.py
python features/e26_multi_encoder.py --encs uni2h,phikon2 --stride 2 --device cuda:0
python training/runner_joint.py --ids j20,j22,j23     # optional: per-encoder screening
python training/e28_ensemble.py                       # optional: pre-set encoder selection rule
python training/train_final_v2.py --device cuda:0     # writes $KNOX_MODEL_DIR/owl_v2/
```

`train_final_v2.py` trains 5 fold models per encoder (folds stratified by cohort x event, seed 2026) and writes
`<encoder>_f<k>.pt` and `calib.json` (per-model and per-encoder z-scoring constants, risk-group quantiles).

Steps that rely on inputs this release does not produce:

- The training tile set is read from `data/feats/virchow2/<ds>/<case>.npz` (key `idx`: indices into the 20x tile grid;
  E0 tissue tiles, uniform stride to at most 4,000). These index files came from an earlier extraction step that is
  not included; only `idx` is used. `features/e26_multi_encoder.py` falls back to the same rule when a file is missing.
- `features/e0_tissue_filter.py` fits on stain-pilot tiles in `data/dict/stain_pilot/` (pixels and tile indices),
  written by a step that is not included.
- The container also needs a `tissue` entry in `owl_v2/calib.json`: the Midnight-only E0 tissue predictor,
  `{"coef": midnight_coef, "intercept": midnight_intercept, "thr": 0.25}` from `data/tissue/predictors.npz`.
  `train_final_v2.py` does not write it; add it after training.
- `features/e7_nuclei.py` imports a nucleus segmentation module `seg` (a distilled Cellpose-SAM U-Net exposing
  `segment(rgb, mpp, device)` that returns a label image). It is not included; point `KNOX_SEG_DIR` at its directory.
- `analysis/e21`, `e23` and `e24` analyse the earlier Midnight-only v1 model (`$KNOX_MODEL_DIR/owl`, 5 ABMIL fold
  models); its training script is not included. The compartment analyses (`e23`, `e24`, `e25`, `e27`, `e31`) use the
  K16 H-optimus-0 dictionary from `features/e2_dictionary.py --tissue --tag _t --Ks 16`.

## Container

Model directory, mounted read-only at `/opt/ml/model` (override with `OWL_MODEL_DIR`):

```
midnight/   Hugging Face snapshot of kaiko-ai/midnight
uni2h/      model.safetensors and timm_kwargs.json for MahmoodLab/UNI2-h
phikon2/    Hugging Face snapshot of owkin/phikon-v2
owl_v2/     calib.json and <encoder>_f<k>.pt from training/train_final_v2.py
```

Build, test and save:

```
cd container
./do_build.sh
KNOX_MODEL_DIR=/path/to/model OWL_INPUT_DIR=/path/to/cases ./do_test_run.sh   # one sub-directory per case
KNOX_MODEL_DIR=/path/to/model ./do_save.sh                                    # image tarball and model.tar.gz
```

`KNOX_MODEL_DIR` defaults to `container/model` and `OWL_INPUT_DIR` to `container/test/input`; outputs are copied to
`container/test/output/<case>/`.

Inputs per case:

- `/input/inputs.json` with the sockets `prostatectomy-tissue-whole-slide-image` and `prostatectomy-tissue-mask`
- `/input/images/prostatectomy-wsi/*.tif` (H&E whole-slide image)
- `/input/images/prostatectomy-tissue-mask/*.tif` (organiser tissue mask)

Outputs per case:

- `/output/overall-survival-years.json`: a ranking time in years, monotone in the negative risk score
- `/output/prostate-cancer-risk-classification.json`: `low`, `medium` or `high` (training-cohort quantiles of the score)
- `/output/images/prostatectomy-overall-survival-attribution-heatmap/attribution.tif`: uint8 map on the mask grid,
  drawn on the patch grid of the organisers' faithfulness test (256 px at 0.5 um/px from the mask origin). Patches
  that overlap a field removed by sequential knockout carry its rank (255 for the first, down to 130); tissue the model
  did not read takes 3 to 129 from the rank of its nearest read field, so the map reads as one picture but always sits
  below every read field (removing unread tissue changes nothing); patches that touch only fields whose removal would
  raise the score are 1, and outside the mask is 0
- `/output/pathology-report.pdf`: five-page report (percentile and band, map, the fields the score rests on, compartment
  and density checks, outcome by band in the development cohorts)

Optional environment variables: `OWL_GPU_MEM_GB` caps GPU memory (the test script sets 16 to emulate a T4),
`OWL_SELECT=list` uses the training tile rule instead of the lattice, and `OWL_DEBUG_DUMP=1` writes per-tile arrays
used by `analysis/e32_mask_edit_e2e.py`.

Mask edits only ever remove analysed fields: the fields are a fixed lattice of grid tiles, and a field leaves the bag
when less than 25% of its window is tissue. If every analysed field has been removed, the development median is
reported; the training tile rule is used only when the tissue is too small for the lattice to hold 16 fields.

Inputs are read defensively: masks as TIFF (any tiling, bit depth or photometric) or MetaImage, with tissue = 1 where
the mask uses 1 and any non-zero value otherwise; the slide's pixel size from its resolution tag, openslide or the
mask's spacing, accepted only between 0.08 and 2 um; a slide region that cannot be read is skipped.

The report flags fields that look like marking ink (an automated colour check on each field read: green, blue or yellow
on at least 20% of the tissue pixels, or near-black on at least 40%), counts them in the quality metrics and states
where the result would move without them. The check is report-only: the prediction and the map do not change.

Evidence regions in the report carry measured nuclear values only; their tissue type is left to the reviewing
pathologist. In a blinded reading of 48 such regions in 24 development cases, three pathologists called 94% carcinoma
(random tissue from the same slides: 42%), and the report states this.

## Citation

Paper under review (WACV 2027 CVPath workshop).

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`.

## Building on Grand Challenge

The root `Dockerfile` builds the same image as `container/Dockerfile` and is what the Grand Challenge GitHub integration uses: link this repository to the algorithm, and every new tag (or GitHub release) triggers a build. Model weights are not in the repository; upload them separately on the algorithm's Models page as a tarball of the model directory (`container/do_save.sh` creates it). At run time they are available at `/opt/ml/model`.

## Faithfulness evaluation

The organisers' notebook (`faithfulness_blackbox.ipynb` in the OWL challenge repository) calls `inference.py` as a
black box through `get_args_parser()` and `run(args)`, one slide and one perturbed mask per call. In the image the code
is at `/opt/app/inference.py` and the model at `/opt/ml/model`. Settings for the notebook's configuration cell:

```python
INFERENCE_PY = Path("/opt/app/inference.py")
MODEL_WSI_DIR = Path("/tmp/faith/wsi")                                   # empty folders that only the notebook fills
MODEL_MASK_DIR = Path("/tmp/faith/mask")
MODEL_OUTPUT_JSON = Path("/tmp/faith/out/overall-survival-years.json")
MODEL_ARGS = ["--wsi-dir", str(MODEL_WSI_DIR), "--mask-dir", str(MODEL_MASK_DIR),
              "--output", str(MODEL_OUTPUT_JSON.parent), "--no-report"]
MODEL_SCRATCH_DIR = None
```

`--no-report` only skips the PDF; the prediction and the map are the same as on Grand Challenge. The attribution map
for `ATTENTION_OUTPUT_DIR` is the algorithm's `attribution.tif` output above. Each call loads the model (about 20 s)
and reads the slide (about 1.5 to 2.5 minutes on one GPU).
