#!/bin/bash
# Since both vosk and fuzzysearch are single-threaded, I wrote this bash script to
# automatically use all 8 processors on my cpu's core. I tried to use GNU make, but
# it doesn't handle spaces in filenames. Like, at all. And all my audiobook files have
# spaces in their filenames. So that's right out. I tried to find another build tool
# that would handle it. Ideally just like make, but supporting spaces.
#
# There's so many options, and so little comparisons, and details are which supports what
# are so hard to find, that I gave up and wrote this bit of stupidity instead. It's not that
# hard, people!
#
# Note that this is a horrible solution, and looks ugly b/c of the multiple "rich console"
# applications that all think they have exclusive access to the console overwriting eachother.
#
if [ "$1" == "" ];then
    echo "Usage: $0 <directory>
    Directory MUST have a .txt file for every .mp3 file."
    exit 1
fi

# My CPU has eight cores.
PROCESSORS=8
for th in "$1"*.txt ;do
    echo thinking about processing "$th"
    while [ `jobs |wc -l` -ge $PROCESSORS ]; do
        echo waiting to start $th
        sleep 10s
    done
    if [ ! -e "${th%txt}csv" ];then
        ./audiobook-to-AI-Training-data.py -s csv "${th%txt}mp3" &
    else
        echo not processing $th, ${th%txt}csv exists!
    fi
done
