# AWS EC2 setup (primary training)

Course guidance: teams receive **$50** AWS credit by default; **$100** possible with an approved written proposal to course staff. Use **EC2** for GPU training.

## Goals

- One reproducible GPU environment for training Img2GPS and exporting **`model.pt`**
- Document enough for the report (**Core Components** — evaluation / reproducibility) without leaking secrets

## Recommended configuration

| Setting | Recommendation | Notes |
|---------|----------------|--------|
| **Service** | Amazon EC2 | “Remote computer” with GPU |
| **AMI** | Deep Learning AMI **or** image with **PyTorch + CUDA** preinstalled | Saves setup time |
| **Instance family** | **g4dn**, **g5**, or **g6** | Cost-effective for student budgets |
| **High-end** | **p** series | Much higher cost; use only if justified |
| **Key pair** | **RSA**, **`.pem`** format | For SSH; **never commit** `.pem` to Git |
| **Storage (EBS)** | Often **> default 8–35 GiB** if storing datasets, conda envs, checkpoints | Pick size for your data + checkpoints |

Fill in when you launch:

- **Region:** TBD (e.g. `us-east-1`)
- **AMI ID / name:** TBD
- **Instance type:** TBD (e.g. `g5.xlarge`)
- **EBS volume (GiB):** TBD
- **Security group:** TBD (restrict SSH to team IPs if possible)

## SSH (example)

```bash
chmod 400 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@<PUBLIC_DNS_OR_IP>
```

(Username may be `ubuntu`, `ec2-user`, or `deep-learning` depending on AMI — check AMI docs.)

## First-time instance setup (after login)

1. Clone this repository (or copy files).
2. Create a virtual environment; install [requirements.txt](../requirements.txt).
3. Verify CUDA if using GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```

4. Sync or download your **training data** (S3, `git lfs`, Hugging Face `huggingface-cli`, etc.) — document the method in [RUNBOOK.md](RUNBOOK.md).

## Cost hygiene

- **Stop or terminate** the instance when not training.
- Optionally log approximate **instance-hours × hourly rate** for transparency (see [RESULTS.md](RESULTS.md)).

## Artifact export

After training, copy **`model.pt`** (and logs) off the instance:

- **SCP** (example): `scp -i key.pem ubuntu@host:/path/model.pt ./artifacts/`
- **S3** or **Hugging Face Hub** — document chosen path in [RUNBOOK.md](RUNBOOK.md)

Record **SHA-256** of the canonical checkpoint in [SUBMISSION_LOG.md](SUBMISSION_LOG.md):

```bash
shasum -a 256 model.pt
```

## Security

- Do not store **Hugging Face tokens** or **AWS access keys** in the repo.
- Use **IAM roles** for EC2 → S3 if applicable, or short-lived credentials on the instance only.
