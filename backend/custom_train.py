#!/usr/bin/env python
"""
Command-line interface voor onze aangepaste Dreambooth trainer die werkt met AITRADER modellen.
"""
import os
import sys

# Bepaal het pad van de huidige script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Voeg de directory toe aan het Python pad
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Importeer onze custom trainer
from custom_trainer import main

if __name__ == "__main__":
    main() 