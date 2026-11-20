#!/usr/bin/env python3
"""Forecast models for the outcome 'sheldus_loss'. See forecast_common.py for what is tested.

    python model_sheldus_loss.py
    python model_sheldus_loss.py --forest --drop t2m
    python model_sheldus_loss.py --reselect --tau-max 6 --alpha 0.05
"""
from forecast_common import run

if __name__ == "__main__":
    run("sheldus_loss")
