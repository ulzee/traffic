#!/bin/bash

for ji in {0..80..10}
do
	screen -X -S j$ji kill
done
