#!/bin/bash

for ji in {0..80..10}
do
	screen -dmS "j$ji" zsh -c "python3 -u dump_many.py s $ji   &>  jobs/job_$ji.log"
done

# python3 dump_many.py s 0   &
# python3 dump_many.py s 10   &
# python3 dump_many.py s 20   &
# python3 dump_many.py s 30   &
# python3 dump_many.py s 40   &
# python3 dump_many.py s 50   &
# python3 dump_many.py s 60   &
# python3 dump_many.py s 70   &
# python3 dump_many.py s 80
