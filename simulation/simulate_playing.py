import pygame
#from pygame.locals import *
from mingus.core import notes, chords
from mingus.containers import *
from mingus.midi import fluidsynth
from os import sys
import os
import time
import sys
sys.path.append('.')
import argparse
import cv2
import numpy as np
import pandas as pd
from trainer.utils import note_idx_to_note

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-data_dir', help='path to data folder', default='/data/neural_hs/simulation_data/raw_exp_data_r06e01mp01kf4', type=str)
args = parser.parse_args()

SF2 = "/usr/share/soundfonts/acoustic_grand_piano_ydp_20080910.sf2" # SF2 file for piano sound
OCTAVES = 5  # number of octaves to show
LOWEST = 2  # lowest octave to show
WHITE_KEYS = ["C", "D", "E", "F", "G", "A", "B"] # set of white keys


# initiate fluidsynth
if not fluidsynth.init(SF2, "alsa"):
    print("Couldn't load soundfont", SF2)
    sys.exit(1)
channel = 0

# initiate pygame piano
# load piano image
pygame.init()
screen = pygame.display.set_mode((640, 480))
image = pygame.image.load("simulation/keys2.png")
(width, height) = (image.get_rect().width, image.get_rect().height)
white_key_width = width / 7

# reset display to wrap around the keyboard image
pygame.display.set_mode((OCTAVES * width, height + 20))
pygame.display.set_caption("Piano")

# pressed is a surface that is used to show where a key has been pressed
pressed = pygame.Surface((white_key_width, height))

# load typed piano notes and model predictions
midis_df = pd.read_csv(os.path.join(args.data_dir, 'midi_data.csv'))
midis_df = midis_df[(midis_df['event'] == 128) | (midis_df['event'] == 144)] # dismiss pedal events
midi_notes = sorted((midis_df.note.value_counts().iloc[0:5].index - 21).tolist())
preds_df = pd.read_csv(os.path.join(args.data_dir, 'preds.csv'), index_col=0) # model predictions

# load presentation of ultrasound images
imgs_names = ['{}.png'.format(x) for x in preds_df.index.values.tolist()]
imgs = [cv2.imread(os.path.join(args.data_dir, 'images', x), cv2.IMREAD_GRAYSCALE) for x in imgs_names]
imgs = [cv2.resize(x, (370,370)) for x in imgs]
cv2.namedWindow('Ultrasound')#, cv2.WINDOW_NORMAL) # opencv window
cv2.moveWindow('Ultrasound', 2135, 592)

# iterate over predictions and execute playing
typed_notes_idxs = np.zeros((5), dtype=np.int32)
for i in range(len(preds_df)):

    # display ultrasound image
    cv2.imshow('Ultrasound', imgs[i])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # clear piano image
    for x in range(OCTAVES):
        screen.blit(image, (x * width, 0))

    # get played notes of frame i
    played_idxs = np.where(preds_df.iloc[i].values > 0)[0].tolist()

    for played_idx in played_idxs:

        # extract note and octave
        note, octave = note_idx_to_note(midi_notes[played_idx])

        # play sound for recently typed note
        if typed_notes_idxs[played_idx] < 1:
            typed_notes_idxs[played_idx] = 1
            fluidsynth.play_Note(Note(note, octave), channel, 100)

        # paint it in the simulation
        octave_offset = (octave - LOWEST) * width
        w = WHITE_KEYS.index(note) * white_key_width
        w = w + octave_offset
        pressed.fill((80, 50, 20))
        screen.blit(pressed, (w, 0), None, pygame.BLEND_SUB)

    for played_idx in range(5):

        # stop sound for recently left note
        if typed_notes_idxs[played_idx] == 1 and preds_df.iloc[i].values[played_idx] == 0:
            typed_notes_idxs[played_idx] = 0
            note, octave = note_idx_to_note(midi_notes[played_idx])
            fluidsynth.stop_Note(Note(note, octave), channel)

    # make sure the frame rate is 19 FPS
    time.sleep(0.048)
    pygame.display.update()
pygame.quit()
