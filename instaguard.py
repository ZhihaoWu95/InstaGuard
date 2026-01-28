import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm
from unet_model import UNet
import lpips
from diffusers import AutoencoderKL
import os
from PIL import Image
import torchvision as tv
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
import time
from fvcore.nn import FlopCountAnalysis

def psnr(img1, img2, max_val=1.0):
    
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val**2 / mse)

def evaluation(args):

    DEVICE = "cuda:{}".format(args.cuda)

    netG = UNet()
    netG = netG.to(DEVICE, dtype=torch.bfloat16)
    netG.eval()
    
    state_dict = torch.load(args.ckpt)
    netG.load_state_dict(state_dict)

    print("Model loaded...")

    transform = transforms.Compose([
    transforms.Resize(args.im_size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),])
    
    perb_single = Image.open('./perb_pattern.png').convert("RGB").resize((args.im_size, args.im_size))
    perb_single = transform(perb_single).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

    victim_path = args.input_path

    save_dir = args.output_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_list = os.listdir(victim_path)
    for image_name in image_list:
        image_path = os.path.join(victim_path, image_name)
        ori_images = Image.open(image_path).convert("RGB")
        ori_images = transform(ori_images).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
        perb = perb_single.repeat(ori_images.size()[0], 1, 1, 1)
        perbed_image = torch.clamp(0.5 * ori_images + 0.5 * perb, -1, 1)
        decode_img = netG(perbed_image)
        save_path = os.path.join(save_dir, image_name)
        tv.utils.save_image(decode_img[0] * 0.5 + 0.5, save_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--input_path', type=str, default='./', help='Input path for the tested images')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for the protected images')
    parser.add_argument('--cuda', type=int, default=1, help='index of gpu to use')
    parser.add_argument('--im_size', type=int, default=512, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    
    args = parser.parse_args()
    evaluation(args)