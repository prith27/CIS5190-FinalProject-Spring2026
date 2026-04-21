# Backend environment (grader parity)

The course **Hugging Face** evaluation environment is restricted to an approved package set. Implementations should run with **these dependencies** unless you have obtained approval via **Ed** for an addition.

## Approved stack (Project_Submission.pdf)

| Package | Notes |
|---------|--------|
| `numpy` | — |
| `pandas` | CSV + labels |
| `torch==2.9.1` | **Pin this version** for local/EC2 parity with the grader |
| `torchvision` | Image transforms / pretrained weights |
| `scikit-learn` | If used |
| `opencv-python` | If used for I/O or transforms |

## Not listed

If you need another library and there is no simple workaround, post on **Ed** and wait for staff approval before relying on it in `model.py` / `preprocess.py`.

## Our development environments

| Environment | Purpose | Verified (date) |
|-------------|---------|-----------------|
| Local (macOS / Linux) | Editing, `eval_project_a.py` | TBD |
| AWS EC2 | Training, checkpoint export | TBD |

## Python version

Record the Python version used on EC2 and locally once verified:

- **Local:** TBD (`python --version`)
- **EC2:** TBD

## Install (example)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

After install, confirm:

```bash
python -c "import torch; print(torch.__version__)"
# expect: 2.9.1
```
