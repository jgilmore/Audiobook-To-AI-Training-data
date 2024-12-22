#!/bin/bash
# Slice files (mp3 or text or whatever, this script assumes the basename is the same)

if [ "$1" == "" -o ! -f "$1" ];then
    echo "Usage: $0 <file with list of filenames, one per line>"
    echo "  calls audiobook-to-AI-Training-data.py <filename> repeatedly."
    echo "  Intended for use AFTER \"Parallel.sh\", to JUST do the slicing."
    exit 1
fi

if [ "$2" != "" ]; then
    if [ ! -x "$1" ]; then
        echo "If to arguments are given, the first MUST be executable!"
        exit 1
    fi
    PROG=$1
    shift
else
    PROG=./audiobook-to-AI-Training-data.py
fi

count=1
while true;do
    file=`head -n $count "$1" | tail -n 1`
    if [ "$file" == "$oldfile" ]; then
        echo "Process finished, slicing ALL DONE!"
    fi
    if "$PROG" "$file"; then
        echo "Successfully processed $file"
    else
        echo "Error: audiobook-to-AI-Training-data.py exited with error $?!"
        echo "File which caused the error was $file"
        echo "Aborting further processing."
        echo "Last successfully processed file was '$oldfile'."
        echo $(( $count - 1 )) files successfully processed
        exit
    fi
    oldfile=$file
    count=$(( $count + 1 ))
done
