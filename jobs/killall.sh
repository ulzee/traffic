#!/bin/bash

for ji in {0..75..15}
do
	screen -X -S j$ji kill
done
