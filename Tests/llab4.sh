#!/bin/bash

rm result.txt

for((i=0;i<=10;i++));   
do   
  for((j=0;j<=10;j++))
  do
    if (($i+$j <= 10));then
      ./llab4 $((1<<$i)) $((1<<$j)) >>"result.txt"
      #nvprof --metrics achieved_occupancy ./llab4 $((1<<$i)) $((1<<$j)) >>"result.txt"
      #nvprof --metrics gld_throughput ./llab4 $((1<<$i)) $((1<<$j)) >>"result.txt" 
      #nvprof --metrics gld_efficiency ./llab4 $((1<<$i)) $((1<<$j)) >>"result.txt"  
    fi
  done  
done
