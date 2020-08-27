#!/bin/bash

echo "It is recommended that this script be run in a python virtual environment."
echo "required packages:  pandas and dcor"

double=1
bindir=$PWD
srcdir=$(dirname "$0")
logfile=input_validation.log

datafile=${srcdir}/../../data/r1000c1000

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -d|--double)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        double=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -b|--bindir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        bindir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -s|--srcdir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        srcdir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -l|--log)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        logfile=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"



echo "running input validation tests using python." > $logfile
#TODO: add commandline parsing.
#TODO: add precision.


for c in csv
do
  for m in 1 2
  do  
    for a in 0 1 2
    do
      # single
      echo "[TEST $c input single core]"
      cmd="${bindir}/bin/${c}_input_test -i ${datafile}.$c -o cpp_input.$c -a $a -m $m"
      echo "$cmd"
      echo "$cmd" >> $logfile
      eval "time $cmd >> $logfile 2>&1"
      python3 ${srcdir}/../python/compare_corr.py double -f ${datafile}.$c -s cpp_input.$c -d ${double}
    done
  done
done


for c in exp
do

  # single
  echo "[TEST $c input single core]"
  cmd="${bindir}/bin/${c}_input_test -i ${datafile}.$c -o cpp_input.$c -a 0 -m 0"
  echo "$cmd"
  echo "$cmd" >> $logfile
  eval "time $cmd >> $logfile 2>&1"
  python3 ${srcdir}/../python/compare_corr.py double -f ${datafile}.$c -s cpp_input.$c -d ${double}

  for m in 2
  do  
    for a in 0 1 2
    do
      # single
      echo "[TEST $c input single core]"
      cmd="${bindir}/bin/${c}_input_test -i ${datafile}.$c -o cpp_input.$c -a $a -m $m"
      echo "$cmd"
      echo "$cmd" >> $logfile
      eval "time $cmd >> $logfile 2>&1"
      python3 ${srcdir}/../python/compare_corr.py double -f ${datafile}.$c -s cpp_input.$c -d ${double}
    done
  done
done
