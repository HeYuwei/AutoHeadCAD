#!/usr/bin/env bash
unzip ${1}/'*.zip' -d data/cq500
cp ${1}/reads.csv data/cq500
