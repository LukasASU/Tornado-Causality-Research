#!/usr/bin/env python3
"""Forecast models for the outcome 'tornado_count'. See forecast_common.py for what is tested.

    python model_tornado_count.py
    python model_tornado_count.py --forest --drop t2m
    python model_tornado_count.py --reselect --tau-max 6 --alpha 0.05
"""
from forecast_common import run

if __name__ == "__main__":
    run("tornado_count")
