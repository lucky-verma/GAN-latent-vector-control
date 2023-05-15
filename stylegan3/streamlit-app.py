### streamlit app to explore the stylegan3 model's latent space

# imports --------------------------------------------------------------
import streamlit as st
import sys
# sys.path.insert(0, "/content/stylegan3")
import pickle
import os
import numpy as np
import PIL.Image
from PIL import Image
from IPython.display import Image
import matplotlib.pyplot as plt
import IPython.display
import torch
import cv2
import dnnlib
import legacy

# config ---------------------------------------------------------------
# streamlit config
st.set_page_config(layout="wide", page_title="GAN latent space explorer", page_icon="🤖")

# hide streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# functions ------------------------------------------------------------
def seed2vec(G, seed):
    return np.random.RandomState(seed).randn(1, G.z_dim)

def display_image(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def get_label(G, device, class_idx):
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class'\
                     'when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running '\
              'on an unconditional network')
    return label

def generate_image(device, G, z, truncation_psi=1.0,
                   noise_mode='const', class_idx=None):
    z = torch.from_numpy(z).to(device)
    label = get_label(G, device, class_idx)
    img = G(z, label, truncation_psi=truncation_psi,
            noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128)\
      .clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

# streamlit app --------------------------------------------------------
st.title("GAN latent space explorer")

# load model
device = torch.device('cuda')
with open("./PlayerGAN.pkl" , 'rb') as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

# sidebar
st.sidebar.title("Latent space controls")
st.sidebar.write("Explore the latent space of the model")

# generate random seed
seed = st.sidebar.slider("Seed", 0, 100_000, 17763)
z = seed2vec(G, seed)

# adjust scale
scale = st.sidebar.slider("Scale", 0.0, 1.0, 0.31)
z = z * scale

# generate image and show in sidebar
truncation_psi = st.sidebar.slider("Truncation psi", 0.0, 1.0, 0.55)
image = generate_image(device, G, z, truncation_psi=truncation_psi)
st.sidebar.image(image, caption="Generated image", width=160)

# Fine-tune ------------------------------------------
st.sidebar.title("Fine-tune")
st.sidebar.write("Fine-tune the latent space of the model")



# main page ------------------------------------------
# explore size
exploresize = st.sidebar.slider("Explore size", 1, 100, 11)
explore = []
for i in range(exploresize):
    # seed numpy random
    np.random.seed(i)
    explore.append(np.random.rand(1, 512) - 0.5)

z = z + explore[-1]

# generate image and show on main page
page_images = []
for c, i in enumerate(explore):
    image = generate_image(device, G, z + explore[c], truncation_psi=truncation_psi)
    # store these images in a list
    page_images.append(image)

# show 10 images per row
for i in range(0, len(page_images), 10):
    st.info("Direction: " + str(range(i, i + 10)))
    st.image(page_images[i:i+10], width=80)


# choose the direction to move
move_direction = st.sidebar.slider("Move direction", 0, exploresize-1, 3)

# if st.sidebar.button("MOVE"):
# show the direction
st.success("Generating images in direction: " + str(move_direction))

z = z + explore[move_direction]

# generate image and show on main page
page_images_move = []
for c, i in enumerate(explore):
    z = z + explore[c]
    image = generate_image(device, G, z, truncation_psi=truncation_psi)
    # store these images in a list
    page_images_move.append(image)

# show 10 images per row
for i in range(0, len(page_images_move), 10):
    st.image(page_images_move[i:i+10], width=80)
