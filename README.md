##### Table of contents

1. [Environment setup](#environment-setup)

2. [Dataset preparation](#dataset-preparation)

3. [How to run InstaGuard](#how-to-run)

# Code implement for InstaGuard

</div>

> **Abstract**: Personalized text-to-image synthesis models (e.g., DreamBooth) can be fine-tuned with a small set of facial images and simple prompts to generate realistic, identity-specific outputs. However, such models pose serious privacy and security risks, as they can be misused to produce harmful or explicit content. Existing protection approaches typically introduce perturbations to facial images, but most rely on gradient-based optimization, which is dependent on the target model, computationally expensive, requires high-end GPUs (e.g., NVIDIA A100), and takes more than ten minutes to complete, thereby severely limiting their practicality for ordinary users in daily protection.
To address these limitations, we propose InstaGuard, a one-step perturbation method that achieves protection with a single, lightweight forward pass and without gradient access. Specifically, we conduct a systematic analysis of existing methods, and extract a common pattern from them. Then, we employ a GAN-based embedding model to efficiently combine the pattern with facial images. InstaGuard completes protection in just 0.4 seconds per image using only 0.36 GB of GPU memory, making it deployable on resource-constrained devices or even CPU hardware. Extensive experiments on VGGFace2 and CelebA-HQ, as well as across various model versions, demonstrate that InstaGuard is effective in preserving personal privacy, maintains high efficiency, and remains robust against image processing and purification.

## Environment setup

Our code relies on the [diffusers](https://github.com/huggingface/diffusers) library from Hugging Face and the implementation of latent caching from [ShivamShrirao's diffusers fork](https://github.com/ShivamShrirao/diffusers).

Install dependencies:

```shell
cd InstaGuard

conda create -n instaguard python=3.9 

conda activate instaguard 

pip install -r requirements.txt 
```

Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from Hugging Face, we use the Stable Diffusion V2.1 as our default model.

## Dataset

We have experimented on these two datasets, you can download from their offcial sources:

- VGGFace2: contains around 3.31 million images of 9131 person identities. We only use subjects that have at least 15 images of resolution above $500 \times 500$.

- CelebA-HQ: consists of 30,000 images at $1024 × 1024$ resolution. 

## Results

For convenient testing, we have provided a split set of one subject in VGGFace2 at `./data/n000050/`.

## How to run InstaGuard

Due to GitHub’s file size limitations, the pre-trained model weights are provided via an anonymous [Huggingface link](https://huggingface.co/AnonymousInstaGuard/InstaGuard/)
 (fully anonymous during the review process). You can download the weights and place them in the ./ckpt folder.

To generate the protected image, you can run

```bash
bash scripts/instaguard_gen.sh
```

</tr>

</table>

If you want to train a DreamBooth model only, you can run:

```bash
bash scripts/train_dreambooth_alone.sh
```

If you want to generate images from the trained models, you can run

```bash
bash scripts/infer.sh
```


