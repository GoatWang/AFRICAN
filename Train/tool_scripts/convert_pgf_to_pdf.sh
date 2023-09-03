#!/usr/bin/bash
filename=${1%.pgf}

# Create the .tex file
echo $filename
echo "\documentclass[preview,border=4mm]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{url}
\usepackage{tikz}
\usepackage{color}
\usepackage{xcolor}
\begin{document}" > tmp.tex
echo "\input{"$filename.pgf"}" >> tmp.tex
echo "\end{document}" >> tmp.tex
echo tmp.tex

# Compile the .tex
pdflatex tmp.tex > cvert.log
mv tmp.pdf $filename.pdf

# # Convert using imagick
# convert -density 500 -units pixelsperinch tmp.pdf $filename.png

# Do some housekeeping
rm -f tmp.tex tmp.aux tmp.pdf tmp.log tmp.fls tmp.fdb_latexmk tmp.ps tmp.dvi cvert.log