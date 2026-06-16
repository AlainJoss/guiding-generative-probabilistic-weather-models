# Data Programming Party Tricks

## SSH Keys

### Check existing keys

```bash
ls -al ~/.ssh
```

### Print public key

```bash
cat ~/.ssh/id_ed25519.pub
```

### Create a new key

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

---

## Git

### Add, commit, and push in one command

```bash
git config --global alias.acp '!f() { git add -A && git commit -m "$*" && git push; }; f'
```

Usage:

```bash
git acp "fix eval pipeline"
```

---

## Bash

### Print current directory inline

```bash
echo "$(pwd)$"
```

---

## Marimo

### Export notebook to PDF

```bash
uv run marimo export pdf notebooks/timestamp_discrepancy.py \
  -o notebooks/timestamp_discrepancy.pdf \
  --include-inputs \
  --include-outputs
```

### Open Marimo config

```bash
nano ~/.config/marimo/marimo.toml
```

### Marimo + Claude setup

```bash
mkdir -p ~/.claude/prompts
curl -L https://docs.marimo.io/CLAUDE.md -o ~/.claude/prompts/marimo.md
npx skills add marimo-team/marimo-pair
```

Use inside a notebook:

```text
/marimo-pair pair with me on my_notebook.py
```

---

## UV

### Upgrade all package versions

```bash
uv lock --upgrade
uv sync
```

---

## Symlink External Data into a Repo

Create a symlink:

```bash
ln -s ../data data
```

Verify:

```bash
ls -ld data
```

Expected:

```text
data -> ../data
```

Remove symlink:

```bash
rm data
```

---

## Long Runs on Mac

### Prevent sleep

```bash
caffeinate -dimsu uv run python script.py
```

### Persistent terminal sessions

Create session:

```bash
tmux new -s rollout
```

Detach:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t rollout
```

---

## GPU Memory Usage

```python
import torch

# device = 0
# props = torch.cuda.get_device_properties(device)

device = torch.device("mps")
props = torch.mps.get_device_properties(device)

print(f"GPU:        {props.name}")
print(f"Total RAM:  {props.total_memory / 1024**3:.2f} GB")
print(f"Allocated:  {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print(f"Reserved:   {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
print(f"Free est.:  {(props.total_memory - torch.cuda.memory_reserved(device)) / 1024**3:.2f} GB")
```

---

## Terminal Commands

* `pwd`: print current directory
* `ls -la`: list files and folders
* `cd`: change directory
* `mkdir`: create folder
* `rm`: delete file or folder
* `cp`: copy file or folder
* `mv`: move or rename file/folder
* `cat`: print file contents
* `grep`: search text in files
* `ssh`: connect to a remote machine
* `scp`: copy files via SSH
* `rsync`: synchronize files efficiently
* `chmod`: change file permissions
* `ps`: list running processes
* `htop`: monitor process monitor
* `kill`: terminate a process
* `wget`: download files from URLs