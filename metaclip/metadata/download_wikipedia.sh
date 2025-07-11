#!/bin/bash

LANG=$1
ROOT=$2
DIR=${ROOT}/${LANG}_text
# if ${LANG}_text/ exists, skip the download
if [ ! -d "$DIR" ] || [ -z "$(ls -A "$DIR")" ]; then
    echo "${ROOT}/${LANG}_text/ does not exist or it's empty. Start the downloading process..."


    file=${LANG}wiki-20240501-pages-articles.xml.bz2

    if [ -f "$file" ]; then
        echo "$file exists."
    else
        echo "Downloading ${LANG}wiki-20240501-pages-articles.xml.bz2"
        wget https://dumps.wikimedia.org/${LANG}wiki/20240501/${LANG}wiki-20240501-pages-articles.xml.bz2
        # if error, abort
        if [ $? -ne 0 ]; then
            echo "Download failed. Exiting..."
            exit 1
        fi
    fi

    echo "Extracting ${LANG}wiki-20240501-pages-articles.xml.bz2 to ${LANG}_text/"
    mkdir -p ${ROOT}/${LANG}_text
    # mkdir -p n-grams

    python -m wikiextractor.WikiExtractor --processes 80 --no-templates ${LANG}wiki-20240501-pages-articles.xml.bz2 -o ${ROOT}/${LANG}_text/
    # if the extraction failed, abort
    if [ $? -ne 0 ]; then
        echo "Extraction failed. Exiting..."
    fi
else
    echo "${ROOT}/${LANG}_text exists. Skip the extracting process..."
fi
