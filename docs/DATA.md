# Data protocol and Hugging Face Dataset

## Overview

TBD: one short paragraph on geographic scope (global vs region) and data source (team capture vs public dataset).

## Collection / curation protocol

TBD:

- How images were gathered or downloaded
- Inclusion/exclusion rules (blur, duplicates, missing GPS)
- License and attribution

## File layout

TBD:

- Directory structure for `train/` / `val/` / `test` (if any)
- CSV column names (must be compatible with grader: `file_name` / `image_path` / `filepath` / `image` / `path` for images; `Latitude` / `latitude` / `lat`; `Longitude` / `longitude` / `lon`)

## Cleaning

TBD:

- Rows removed and why
- EXIF failures or corrupt files

## Splits

| Split | Rows | Notes |
|-------|------|--------|
| Train | TBD | |
| Validation | TBD | |
| Test (if any) | TBD | |

TBD: random vs geographic / temporal holdout; leakage risks and mitigations.

## Final dataset statistics

TBD:

- Approximate geographic coverage (map or bbox)
- Class balance N/A (regression); optional histogram of lat/lon density

## Hugging Face Dataset

**URL:** TBD — required in the course report per Project_Submission.pdf.

**Dataset card:** TBD (description, license, citation).

## Known limitations

TBD: bias, label noise (consumer GPS), indoor vs outdoor, etc.
