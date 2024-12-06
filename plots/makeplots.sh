#!/usr/bin/env sh

python3 makeplot1.py
python3 makeplot2.py ./data/Karlen_data.csv ./eps/plot2
python3 makeplot2.py ./data/Karlen_data_2.csv ./eps/plot3
python3 makeplot2.py ./data/Rutledge_data.csv ./eps/plot4

for f in ./eps/*; do 
    base=$(basename -s .eps $f)
    gs -sDEVICE=pdfwrite -dEPSCrop -o pdf/$base.pdf $f
done
