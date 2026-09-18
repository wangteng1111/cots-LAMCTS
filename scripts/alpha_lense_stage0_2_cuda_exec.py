#!/usr/bin/env python3
"""Exec Alpha Lense Stage0-2 with the workstation's validated CUDA venv."""
import os,sys
py="/var/lib/cots-lamcts/venv/bin/python"
script=os.path.join(os.path.dirname(os.path.dirname(__file__)),"scripts","alpha_lense_stage0_2_run.py")
os.execv(py,[py,"-u",script,*sys.argv[1:]])
