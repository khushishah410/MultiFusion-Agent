# Dataset guide

Research done on 2026-10-27 from official pages, the PhysioNet project pages, the BIMCV agreement, published papers and Academic Torrents listings. Re-check terms before use; they change.

## Comparison

| Dataset (id) | Official source | Modalities | Documented size / counts | Access | Download in this project | Intended use here |
|---|---|---|---|---|---|---|
| IU X-Ray / Open-i (`iu_xray`) | https://openi.nlm.nih.gov/faq.php | PNG radiographs, XML reports (indication, findings, impression), MeSH terms | Reports archive 1.11 MB, PNG archive 1.36 GB (Academic Torrents listing); 7,470 images, 3,955 reports (Babar et al. 2021, PMC8629217) | Public download from NLM; third-party listings state CC BY-NC-ND 4.0 | **Automatic** (`https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz`, `.../NLMCXR_png.tgz`) | Primary integration: reports + images + weak labels |
| NIH ChestX-ray14 (`nih_cxr14`) | https://nihcc.app.box.com/v/ChestXray-NIHCC | PNG, NLP-mined labels, age/sex/view | ChestX-ray8 paper: 108,948 images, 32,717 patients, 8 labels; the CXR14 release extends to 14 labels | No use restrictions; link to NIH site, cite Wang et al. 2017, acknowledge NIH CC (Google Cloud dataset page) | Manual import (Box share links are not a stable API) | Labels without reports |
| CheXpert (`chexpert`) | https://aimi.stanford.edu/datasets/chexpert-chest-x-rays | JPG, labeler labels incl. uncertain (-1) | 224,316 radiographs, 65,240 patients (Stanford AIMI) | Registration + research use agreement | Manual import | Uncertain-label handling |
| MIMIC-CXR-JPG 2.1.0 (`mimic_cxr_jpg`) | https://physionet.org/content/mimic-cxr-jpg/2.1.0/ | JPG, CheXpert/NegBio labels, official splits | 377,110 images; 227,827 labelled studies | Credentialed PhysioNet + CITI training + DUA | Manual import | Large-scale evaluation |
| VinDr-CXR 1.0.0 (`vindr_cxr`) | https://physionet.org/content/vindr-cxr/1.0.0/ | DICOM, radiologist boxes, 28 labels | 18,000 images (15,000 train / 3,000 test) | Credentialed PhysioNet + DUA | Manual; DICOM→PNG conversion is **not** included | Expert labels |
| PadChest (`padchest`) | https://bimcv.cipf.es/?p=4923 | PNG, labels from Spanish reports | >160,000 images, ~67,000 patients; 27% of reports manually labelled (Bustos et al.) | BIMCV research use agreement; no redistribution | Manual import | Multilingual labels |
| Synthetic demo (`synthetic_demo`) | generated locally | gradient PNGs stamped "SYNTHETIC", placeholder text | 12 records | none | `scripts.generate_demo_data` | Software testing only |

Not verified here: total sizes for NIH, CheXpert, MIMIC, VinDr and PadChest (their pages did not state them in the parts read). No dataset publishes checksums that this project could find, so downloads are verified by HTTP status, content type, `Content-Length` and a locally recorded SHA-256.

## Why IU X-Ray is the primary dataset

It is the only candidate with (a) a direct, unauthenticated download from the publisher, (b) radiology report text (indication, findings, impression) that matches the multimodal goal, (c) a size practical on a laptop (1.36 GB), and (d) a metadata-only option (1.1 MB). Its weaknesses: labels are MeSH headings (weak, positives only), there are no patient ids, and the TorchXRayVision `all` weights were trained on it (see MODEL_GUIDE.md).

### Mirrors and integrity (IU X-Ray)

Sources are tried in order: the official Open-i URL, then the web seeds listed in the Academic Torrents `.torrent` metadata for the same archives (archive.org, academictorrents.cloud.seedboxes.cc, host1.academictorrents.com, and for the images academictorrents1.cfhcable.net). The status file records which source served each file.

| Archive | Exact size | Check |
|---|---|---|
| `NLMCXR_reports.tgz` | 1,112,632 bytes | exact size only (the `.torrent` copy available while building was damaged: its hash bytes had been replaced with U+FFFD) |
| `NLMCXR_png.tgz` | 1,360,814,128 bytes | exact size + 325 SHA-1 piece hashes from `config/torrents/NLMCXR_png.tgz.torrent`, infohash `5a3a439df24931f410fac269b87b050203d9467d` (matches the Academic Torrents id) |

A torrent match shows the file is identical to the archive the torrent's creator published in November 2018 (listing Open-i as its source). It does not prove the copy matches what NLM serves today, because NLM publishes no checksums. `dataset_manager verify --dataset iu_xray --torrent <file>` accepts an intact `.torrent` for the reports archive.

**Acquisition status in the build environment:** the download code ran, but `openi.nlm.nih.gov` and every mirror were blocked by that environment's network policy (HTTP 403 at the proxy or DNS failure). The failure was recorded as `failed`, not as success. No IU X-Ray file was downloaded or validated there. The IU parser was tested on fixture XML that follows the structure used by published loaders (`pmcId`, `uId`, `MedlineCitation/Article/Abstract/AbstractText[@Label]`, `MeSH/major`, `parentImage`); run `prepare` + `validate` after downloading on your machine and check the build issues it prints.

## Workflow

```
download/import ─▶ data/raw/<id>/ (or your folder, read-only)
prepare        ─▶ data/manifests/<id>.jsonl + <id>.meta.json
validate       ─▶ <id>.validation.json + <id>.validation.md
activate       ─▶ data/manifests/active.json (pointer used by the API)
```

`data/raw/<id>/.dataset_status.json` records each file's URL, byte count, SHA-256, timestamp, resume flag and extracted file count, plus the overall state (`not_started`, `in_progress`, `metadata_only`, `complete`, `imported_local`, `failed`) and the last error.

Safety rules in the code:
- Downloads stream to `<name>.part` in 1 MB chunks and resume with HTTP Range when the server supports it; a server that ignores the range restarts cleanly.
- HTML responses (login pages) are rejected; incomplete files stay as `.part`.
- Free disk space is checked (10% headroom) before writing.
- Archives: every member is checked for absolute paths, `..`, links and special files before extraction; nothing downloaded is executed.
- Downloads above 200 MB (setting `large_download_mb`) need `--yes` or the UI confirmation box.
- Gated datasets have no download code at all; the `import` command only registers a folder you obtained.

## Normalised manifest

One JSON object per line. Every field except `case_id` and `dataset_source` is optional.

| Field | Type | Meaning |
|---|---|---|
| `case_id` | str | Unique per manifest (trimmed; original kept in `metadata.source_case_id`) |
| `patient_id` | str? | Grouping key for splits. Sensitive: never returned by the API |
| `clinical_indication` | str? | Reason for the exam |
| `symptoms` | list[str] | Structured symptoms if the source has them |
| `radiology_findings`, `radiology_impression` | str? | Report sections |
| `image_ids` | list[str] | Source image identifiers |
| `image_paths` | list[str] | Relative to the dataset root; aligned with `image_ids`; expected path kept even if the file is missing |
| `labels` | dict[str,int] | 1 positive, 0 negative, exactly as the source states |
| `label_uncertainty` | list[str] | Labels the source marks uncertain (CheXpert -1); never converted |
| `label_source` | str? | How labels were produced |
| `mesh_major` | list[str] | IU MeSH major terms, raw |
| `dataset_source` | str | Dataset id |
| `split` | str? | `train`/`val`/`test` |
| `synthetic` | bool | True only for generated records |
| `metadata` | dict | Source-specific extras (comparison, MeSH minor, view position, raw labels...) |

Sample (IU X-Ray structure):

```json
{"case_id": "CXR1", "patient_id": null, "clinical_indication": "<INDICATION section text>",
 "radiology_findings": "<FINDINGS section text>",
 "radiology_impression": "<IMPRESSION section text>", "image_ids": ["CXR1_1_IM-0001-3001", "CXR1_1_IM-0001-4001"],
 "image_paths": ["CXR1_1_IM-0001-3001.png", "CXR1_1_IM-0001-4001.png"], "labels": {},
 "label_uncertainty": [], "label_source": "MeSH major headings of the report (weak, positives only)",
 "mesh_major": ["normal"], "dataset_source": "iu_xray", "split": "train", "synthetic": false,
 "metadata": {"source_file": "ecgen-radiology/1.xml", "labels_are_positives_only": true, "mesh_normal": true}}
```

Angle-bracket values are placeholders; ids and file names follow the IU naming pattern.

## Builders

| Builder | Reads | Notes |
|---|---|---|
| `iu_xray_xml` | `**/*.xml`, images by file stem | Labels = MeSH heading before `/`, except "normal"; positives only |
| `nih_csv` | `Data_Entry_2017*.csv`, optional `train_val_list.txt`/`test_list.txt` | All 14 labels set 0/1; official test list respected; train/val split by patient |
| `chexpert_csv` | `train.csv`, `valid.csv` | Blank = not mentioned (omitted); -1 → `label_uncertainty`; patient id from path |
| `mimic_csv` | `mimic-cxr-2.0.0-chexpert.csv(.gz)`, `-split.csv(.gz)` | One case per study; official split |
| `generic` | `.jsonl`/`.json`/`.csv` with normalised names, or `--field-map map.yaml` | Lists may be JSON or `|`-separated |

## Splits and leakage

Source splits are kept. Otherwise records are split 70/15/15 by a hash of `patient_id` (or `case_id` when no patient id exists, as in IU X-Ray, so two reports of one IU patient could land in different splits; this is reported as the grouping key). Validation reports groups or images that appear in more than one split.

## Path handling

All image paths go through `safe_join`: absolute paths, drive letters and `..` are rejected, and the resolved path must stay inside the dataset root. The API serves images only via `/dataset/cases/{id}/images/{n}` and never returns absolute paths.

## Storage

IU X-Ray needs about 1.4 GB for the archives plus the same again extracted; you may delete `NLMCXR_png.tgz` after a successful `prepare` (`dataset_manager cleanup` prints the paths). `scripts.prepare_images --dataset iu_xray --size 512` writes smaller copies to `data/processed/` without touching originals.
