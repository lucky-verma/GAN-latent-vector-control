### streamlit app to display stylemixing on a trained stylegan2 model

# imports --------------------------------------------------------------
import os
import streamlit as st
import re
from typing import List
import legacy
import time

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

# config ---------------------------------------------------------------
# streamlit config
st.set_page_config(layout="wide", page_title="GAN Style Mixing", page_icon="🤖")

# hide streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

placeholder = st.empty()

# load networks from pkl file
network_pkl = 'stylegan_human_v2_1024.pkl'

placeholder.info('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with open(r"D:\WORK\Personal\GAN-latent-vector-control\stylegan3\pages\network-snapshot-001160.pkl", 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)


placeholder.info('Generating W vectors...')


# streamlit to get rows and cols seed values
row_seeds = st.sidebar.text_input('Enter row seed values separated by commas:', ','.join([str(x) for x in [*range(1, 6)]])).split(',')
col_seeds = st.sidebar.text_input('Enter column seed values separated by commas:', ','.join([str(x) for x in [*range(6, 30)]])).split(',')
row_seeds = [int(x) for x in row_seeds]
col_seeds = [int(x) for x in col_seeds]

# options to sample 5 random seeds for rows and cols
if st.sidebar.button('Sample random seeds'):
    row_seeds = np.random.randint(1000, size=5)
    col_seeds = np.random.randint(1000, size=25)
    row_seeds = [int(x) for x in row_seeds]
    col_seeds = [int(x) for x in col_seeds]
    st.sidebar.info('Row seeds: %s' % str(row_seeds))
    st.sidebar.info('Col seeds: %s' % str(col_seeds))


# streamlit to get truncation psi value
truncation_psi = st.sidebar.slider('Truncation psi value:', 0.0, 1.0, 0.7, 0.1)

all_seeds = list(set(row_seeds + col_seeds))
all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
w_avg = G.mapping.w_avg
all_w = w_avg + (all_w - w_avg) * truncation_psi
w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

# streamlit to get noise mode
noise_mode = st.sidebar.selectbox('Noise mode:', ['const', 'random', 'none'])

placeholder.info('Generating images...')
all_images = G.synthesis(all_w, noise_mode=noise_mode)
all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

# streamlit to select layers to mix
col_styles = st.sidebar.multiselect('Select layers to mix:', list(range(18)), default=[0, 1, 2, 3, 4, 5, 6, 7])

placeholder.info('Generating style-mixed images...')
with st.empty():
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()
            placeholder.success('Generating style-mixed images... %s/%s' % (row_seed, col_seed))

placeholder.info('Generating style-mixed images... Done!')
W = G.img_resolution // 2
H = G.img_resolution
canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
for row_idx, row_seed in enumerate([0] + row_seeds):
    for col_idx, col_seed in enumerate([0] + col_seeds):
        if row_idx == 0 and col_idx == 0:
            continue
        key = (row_seed, col_seed)
        if row_idx == 0:
            key = (col_seed, col_seed)
        if col_idx == 0:
            key = (row_seed, row_seed)
        canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
st.image(canvas, use_column_width=True)
