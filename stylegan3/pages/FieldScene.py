### streamlit app to generate players that are placed on a random positions on a football field

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
import PIL.ImageDraw
import torch

# config ---------------------------------------------------------------
# streamlit config
st.set_page_config(layout="wide", page_title="Scene", page_icon="🤖")

# hide streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

placeholder = st.empty()
placeholder = st.empty()

# load networks from pkl file
network_pkl = 'stylegan_human_v2_1024.pkl'

placeholder.info('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with open(r"..\stylegan3\pages\network-snapshot-001160.pkl", 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
placeholder.info('Generating W vectors...')


# display the field image from assets folder using relative path
field_image = PIL.Image.open(os.path.join('assets', 'field_crop.jpeg'))
# st.image(field_image, use_column_width=True)

# display 4 points ------------------------------------------------------

# write the 4 points on the field image
# point 1
point1_x = 1064
point1_y = 471
# point 2
point2_x = 336
point2_y = 939
# point 3
point3_x = 3424
point3_y = 935
# point 4
point4_x = 2735
point4_y = 474

# draw a polygon on the field image using the 4 points
field_image_polygon = field_image.copy()
# create a polygon object
polygon = PIL.ImageDraw.Draw(field_image_polygon)
polygon.line([(point1_x, point1_y), (point2_x, point2_y), (point3_x, point3_y), (point4_x, point4_y), (point1_x, point1_y)], fill='red', width=5)

# display the polygon image
st.image(field_image_polygon, "Metlife Stadium",use_column_width=True)

# get random positions for the players ------------------------------------
placeholder.info('Generating positions for players...')
# x positions for team 1
x1_positions = np.random.randint(point1_x, point4_x, size=11)
# y positions for team 1
y1_positions = np.random.randint(point1_y, point2_y, size=11)

# x positions for team 2
x2_positions = np.random.randint(point1_x, point4_x, size=11)
# y positions for team 2
y2_positions = np.random.randint(point1_y, point2_y, size=11)

# display the players on the field image
field_image_players = field_image.copy()
# create a polygon object
polygon = PIL.ImageDraw.Draw(field_image_players)
# draw the players on the field
for i in range(11):
    # team 1
    polygon.ellipse([(x1_positions[i]-10, y1_positions[i]-10), (x1_positions[i]+10, y1_positions[i]+10)], fill='red', outline='red')
    # team 2
    polygon.ellipse([(x2_positions[i]-10, y2_positions[i]-10), (x2_positions[i]+10, y2_positions[i]+10)], fill='blue', outline='blue')

# display the polygon image
st.image(field_image_players, "Metlife Stadium",use_column_width=True)

# sidebar to get player generation  ------------------------------------


# Sample players using seed as x and y coordinates
# get seed values for players on team 1 using x and y coordinates
x1_seed = st.sidebar.text_input('Enter x1 seed values separated by commas:', ','.join([str(x) for x in x1_positions])).split(',')
y1_seed = st.sidebar.text_input('Enter y1 seed values separated by commas:', ','.join([str(x) for x in y1_positions])).split(',')
x1_seed = [int(x) for x in x1_seed]
y1_seed = [int(x) for x in y1_seed]

# get seed values for players on team 2 using x and y coordinates
x2_seed = st.sidebar.text_input('Enter x2 seed values separated by commas:', ','.join([str(x) for x in x2_positions])).split(',')
y2_seed = st.sidebar.text_input('Enter y2 seed values separated by commas:', ','.join([str(x) for x in y2_positions])).split(',')
x2_seed = [int(x) for x in x2_seed]
y2_seed = [int(x) for x in y2_seed]


if st.sidebar.button('Generate images'):
    # select first value of y1 and y2 as the row seeds
    # select all the values of x1 and x2 as the col seeds 
    row_seeds = [y1_seed[0], y2_seed[0]]
    col_seeds = x1_seed + x2_seed
    
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * 0.7 # truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    placeholder.info('Generating images...')
    all_images = G.synthesis(all_w, noise_mode='const') # ['const', 'random', 'none']
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
                image = G.synthesis(w[np.newaxis], noise_mode='const') # ['const', 'random', 'none']
                image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()
                placeholder.success('Generating style-mixed images... %s/%s' % (row_seed, col_seed))

    placeholder.info('Generating style-mixed images... Done!')
    
    # get the team 1 and team 2 images
    team1 = [image_dict[(row_seeds[0], col_seed)] for col_seed in col_seeds[:11]]
    team2 = [image_dict[(row_seeds[1], col_seed)] for col_seed in col_seeds[11:]]

    # display the teams by concatenating the images
    st.image(np.concatenate(team1, axis=1), "Team 1",use_column_width=True)
    st.image(np.concatenate(team2, axis=1), "Team 2",use_column_width=True)

    # scale the players with respect to the field image

    avg_player_height = 72
    field_width = 3600
    field_height = 1920




    # Overlay the team1 and team2 images on the field image with the respective x and y coordinates
    field_image_overlay = field_image.copy()
    
    # paste the team1 images on the x1 and y1 coordinates
    for i in range(11):
        field_image_overlay.paste(PIL.Image.fromarray(team1[i]), (x1_positions[i]-64, y1_positions[i]-64))

    # paste the team2 images on the x2 and y2 coordinates
    for i in range(11):
        field_image_overlay.paste(PIL.Image.fromarray(team2[i]), (x2_positions[i]-64, y2_positions[i]-64))

    # display the polygon image
    st.image(field_image_overlay, "Metlife Stadium",use_column_width=True)


