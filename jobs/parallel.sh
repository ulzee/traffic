#!/bin/bash

for ji in {0..75..15}
do
	screen -dmS "j$ji" zsh -c "python3 -u $1 $ji $(($ji + 15))   &>  jobs/job_$ji.log"
	# screen -dmS "j$ji" zsh -c "python3 -u dump_many.py s $ji   &>  jobs/job_$ji.log"
done
