#!/bin/sh

echo 'Correlation between "OnWN" train data'
./correlation-noconfidence.pl data/STS.gs.OnWN.txt out/STS.result.OnWN.txt

echo 'Correlation between "images" test data'
./correlation-noconfidence.pl data/STS.gs.images.txt out/STS.result.images.txt
