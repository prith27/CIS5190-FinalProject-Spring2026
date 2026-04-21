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
| **Key pair** | **RSA**; use **Import key pair** (see below) if **Create** is blocked by org policy | Private key **never** in Git |
| **Storage (EBS)** | Often **> default 8–35 GiB** if storing datasets, conda envs, checkpoints | Pick size for your data + checkpoints |

Fill in when you launch:

- **Region:** TBD (e.g. `us-east-1`)
- **AMI ID / name:** TBD
- **Instance type:** TBD (e.g. `g5.xlarge`)
- **EBS volume (GiB):** TBD
- **Security group:** TBD (restrict SSH to team IPs if possible)

---

## Step-by-step: Launch a GPU instance (AWS console)

Use this once your **AWS account** is active and any **course / Educate** credits are applied (billing → Credits). All steps are in the **AWS Management Console** in a browser.

### 1. Open EC2 and pick a region

1. Sign in to [https://console.aws.amazon.com/](https://console.aws.amazon.com/).
2. Top-right **Region** menu: choose one with GPU capacity and good pricing for you (common: **US East N. Virginia** `us-east-1`). Stay in that region for every step below.

### 2. SSH key pair (course default: import from your laptop)

Some Penn / org accounts **deny** `ec2:CreateKeyPair` in the console. Course staff recommend **generating the key on your machine** and **importing the public key** (same security model; you keep the private key locally).

1. **On your laptop** (macOS / Linux), generate a key (**PEM** private key format):

   ```bash
   ssh-keygen -t rsa -b 4096 -m PEM -f ~/.ssh/cis5190_aws
   ```

   This creates:
   - **Private (secret):** `~/.ssh/cis5190_aws` — used only for `ssh` / `scp` (**never** commit, never email).
   - **Public:** `~/.ssh/cis5190_aws.pub` — this is what you upload to AWS.

2. In the AWS console: **EC2** → **Key Pairs** → **Import key pair** → name e.g. `cis5190-key` → upload **`cis5190_aws.pub`**.

3. If **Import** is also denied, contact course staff with the error.

**Alternative (only if the console allows it):** **Create key pair** in EC2, type **RSA**, **.pem** — download the `.pem` once; you cannot re-download.

### 3. Launch the instance

1. **EC2** → **Instances** → **Launch instances**.
2. **Name** (optional): e.g. `img2gps-train`.
3. **Application and OS Images (AMI)** — pick **one** of:
   - **Ubuntu** + **Deep Learning AMI** (search “Deep Learning” or “Ubuntu DL” in AMI search), **or**
   - **Ubuntu Server 22.04 LTS** (you will `pip install torch` with CUDA wheels yourself — a bit more setup).
   - *Tip:* A **Deep Learning Base** or **PyTorch** AMI already has NVIDIA drivers/CUDA; still verify with `nvidia-smi` after boot.
4. **Instance type** → **Browse more types** → filter **GPU** → choose e.g. **`g5.xlarge`** or **`g4dn.xlarge`** (cheaper, older GPU). Avoid **`p`** instances unless you know you need them (cost).
5. **Key pair** → select the **imported** key (e.g. `cis5190-key`) or the one you created in AWS.
6. **Network settings** → **Edit**:
   - Allow **SSH (22)** from **My IP** (recommended) or a known IP range — **not** `0.0.0.0/0` unless you accept the risk.
7. **Configure storage** → **Edit**:
   - Default root volume is often **too small** for HF cache + Docker/conda + checkpoints. Use at least **80–100 GiB** **gp3** for Img2GPS dataset build + training (more if you materialize large caches).
8. **Advanced details** (optional): nothing required for a first launch; IAM role can be added later if you use S3.
9. **Launch instance** → wait until **Instance state** is **Running**.

### 4. Connect with SSH

1. **EC2** → **Instances** → select your instance → copy **Public IPv4 DNS** (or IPv4 address).
2. On your laptop (terminal):

   ```bash
   chmod 400 ~/.ssh/cis5190_aws
   ssh -i ~/.ssh/cis5190_aws ubuntu@<PUBLIC_DNS_OR_IP>
   ```

   (If you used a downloaded **.pem** from **Create key pair**, point `-i` at that file instead.) If login fails, try user **`ec2-user`** (Amazon Linux) or check the AMI’s **Connect** tab for the exact username.

3. On the instance, confirm GPU:

   ```bash
   nvidia-smi
   ```

### 5. After login (Img2GPS)

Continue with **First-time instance setup** below: clone repo, venv, `pip install -r requirements.txt`, Hugging Face login, then [RUNBOOK.md](RUNBOOK.md) (dataset build / training).

### 6. Cost and cleanup

- **Stop** the instance when idle (you stop paying for the instance; EBS storage still bills unless you delete the volume).
- **Terminate** when fully done if you don’t need the disk; **terminate deletes the instance’s root volume** unless you snapshot it.
- Optional: **Elastic IP** if you need a stable public IP across stop/start (avoid leaving an unattached Elastic IP — it can incur small charges).

---

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

## Troubleshooting: cannot create a key pair (`ec2:CreateKeyPair` denied)

If **Create key pair** is blocked by an org **SCP**, use **Import key pair** with a key generated on your laptop — see **Step 2** above (`ssh-keygen` … `-m PEM`, upload **`.pub`**). That is the **supported** flow for many Penn accounts.

If **Import key pair** is also denied with the **same org SCP** as `CreateKeyPair`, **both** API calls are blocked — you **cannot** register a key in EC2 for that account. Only **course / cloud admins** can fix this (policy change, different account, or an official connect path). Email staff with both error messages. On the **instance → Connect** page, you can still check **EC2 Instance Connect** or other tabs in case your role allows access without a key pair, but that depends on the org.

## Security

- Do not store **Hugging Face tokens** or **AWS access keys** in the repo.
- Use **IAM roles** for EC2 → S3 if applicable, or short-lived credentials on the instance only.
