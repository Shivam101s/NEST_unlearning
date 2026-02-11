# NEST: Neuron-Level Selective Unlearning for Foundation Models

Official PyTorch implementation of **NEST**, a training-free, neuron-level machine unlearning framework for removing targeted concepts or identities from deep models while preserving utility on retained data.

This repository accompanies our research work on efficient and scalable unlearning in foundation models.

---

## Overview

Machine unlearning aims to remove specific information from trained models without full retraining.  
**NEST** achieves this by:

- Identifying concept-relevant neurons
- Applying statistically controlled masking
- Performing one-shot selective updates
- Preserving retained knowledge via structure-aware constraints

### Key Features

- Training-free unlearning
-  Neuron-level precision
-  FDR-controlled selection
-  Works with CLIP / Vision-Language Models
-  Efficient and scalable
-  Minimal performance degradation

---


### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/NEST_unlearning.git
cd NEST_unlearning

pip install -r requirements.txt

To calculate CLIP important neuorns: neuron_importance_clip.py
To unlearn CLIP: src/clip/unlearn_clip_nest.py

Important neurons for SD: sd_neuron_importance.py
Unlearn SD: sd_unlearn.py

VLM_unlearning: neuron_vlm_unlearn.py
VLM inference: inference_vlm.py


