#!/bin/bash
CACHE=$1
python run_two.py checkpoint=outputs/AllegroHandHora/${CACHE}/stage2_nn/best.pth
