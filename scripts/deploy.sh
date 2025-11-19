#!/bin/bash
CACHE=$1
python run.py checkpoint=outputs/RightAllegroHandHora/${CACHE}/stage2_nn/best.pth
