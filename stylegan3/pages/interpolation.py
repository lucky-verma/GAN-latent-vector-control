### streamlit app to perform interpolation on a trained stylegan2 model

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
st.set_page_config(layout="wide", page_title="GAN Interpolation", page_icon="🤖")

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