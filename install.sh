#!/bin/bash
pip install -r requirements.txt

cd packages/nnunetv2
pip install -e .
    
cd ../report-guided-annotation
pip install -e .