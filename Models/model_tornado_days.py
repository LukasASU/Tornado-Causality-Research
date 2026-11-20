#!/usr/bin/env python3
"""Forecast models for the outcome 'tornado_days'. See forecast_common.py for what is tested.

    python model_tornado_days.py
    python model_tornado_days.py --forest --drop t2m
    python model_tornado_days.py --reselect --tau-max 6 --alpha 0.05
"""
from forecast_common import run

if __name__ == "__main__":
    run("tornado_days")
