#!/bin/bash
# monitor CPU and memory usage


TotalMem=$(free -m -t | awk '/Mem:/ {print $2}')

while :
do
    read -r PCPU PMEM <<< $(ps -ef -o user,%cpu,%mem,cmd | grep llm_retrievalqa_gguf.py | head -n 1 | sed "s/\s\s*/ /g" | cut -d " " -f 2,3)
    UsedMem=$(bc <<< "$TotalMem * $PMEM / 100")

    echo "%CPU: $PCPU%"
    echo "%MEM: $PMEM%"
    echo "MEM: $UsedMem MB"
    echo "==================="

    sleep 2
done
