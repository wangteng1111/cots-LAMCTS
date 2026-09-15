#!/usr/bin/env python3
"""Bridge-safe Alpha Lense Optical Bench collector entrypoint."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from alpha_lense.data.collect_opticalbench import main
if __name__=='__main__':
    main()
