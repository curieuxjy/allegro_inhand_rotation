#!/bin/bash
CACHE=$1
python run.py checkpoint=outputs/AllegroHandHora/${CACHE}/stage2_nn/best.pth
