#!/bin/bash
mkdir -p lessons/results
mkdir -p lessons/figures

# Download Data
curl -L "https://www.dropbox.com/scl/fi/gfuo9ywg1hmz5ozuh5ga8/data.zip?rlkey=8tpxtcmvgpi2xwweql8bnnrbu&dl=1" -o lessons/data.zip
unzip lessons/data.zip -d lessons
rm lessons/data.zip
rm -rf lessons/__MACOSX