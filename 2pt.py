#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 zyqiao
# All rights reserved.

from absEntropy import *
import argparse as ap
import time
from rich.console import Console

def main():
    show_kanban()
    params = ap.ArgumentParser(description='Two-Phase Thermodynamic Model for the absolute entropy in mixtures.')
    params.add_argument('-c', '--configure', type=str, help='configuration yaml file for the simulation and groups')
    args = params.parse_args()

    Groups = AtomGroups(args.configure)
    Groups.remove_groups()
    model_2pt = DOSdist(args.configure, Groups)
    # work flow
    model_2pt.print_information()
    model_2pt.velocity_analysis()
    model_2pt.vacf()
    model_2pt.dos()
    model_2pt.dos_2pt()
    model_2pt.vibration_analysis()
    model_2pt.save_data()

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()

    console = Console()
    console.rule(f"Task Finished, cost {(time_end - time_start)/60:6.2f} mins")
