#!/bin/bash

args=""
for arg in $@; do
    if [[ $arg =~ "--" ]]; then
        args="$args $arg"
    else
	args="$args=$arg"
    fi
done

SAT_COUNT=0
UNSAT_COUNT=0
TOTAL_COUNT=0
TOTAL_TIME=0
base_path='/home/ma-user/SatBenchmark/optimize/logs/EasyNaS/kissat'         ######
result_path="${base_path}/kissat2_result.txt"       ######

KISSAT_EXEC="/home/ma-user/SatBenchmark/optimize/kissat-rel-4.0.1/build/kissat ${args}"        ######
echo ${KISSAT_EXEC} | tee ${result_path}

mkdir -p ../SAT_result
mkdir -p "${base_path}/logs/"

for file in /home/ma-user/SatBenchmark/test_data/sample_12sat/*.cnf; do     ######
    FILENAME=$(basename $file)
    echo "Solving ${FILENAME}..."
    LOGFILE="${base_path}/logs/$FILENAME.log"
    TIME_START=$(date +%s.%N)
    $KISSAT_EXEC $file | tee $LOGFILE
    RESULT=$(grep -E "s SATISFIABLE|s UNSATISFIABLE" $LOGFILE)
    TIME_END=$(date +%s.%N)
    TIME_TAKEN=$(echo "$TIME_END - $TIME_START" | bc)
    echo "$FILENAME $RESULT Time Taken: $TIME_TAKEN seconds" | tee -a ${result_path}
    TOTAL_TIME=$(echo "${TOTAL_TIME} + ${TIME_TAKEN}" | bc)
    if [[ $RESULT == *"SATISFIABLE"* ]]; then
        ((SAT_COUNT++))
    elif [[ $RESULT == *"UNSATISFIABLE"* ]]; then
        ((UNSAT_COUNT++))
    fi
    ((TOTAL_COUNT++))
    if (( $TOTAL_COUNT % 5 == 0 )); then
        SAT_RATIO=$(echo "$SAT_COUNT $TOTAL_COUNT" | awk '{print $1 / $2 * 100}')
        # UNSAT_RATIO=$(echo "$UNSAT_COUNT $TOTAL_COUNT" | awk '{print $1 / $2 * 100}')
        echo "After $TOTAL_COUNT tests: SATISFIABLE: $SAT_RATIO%, COSTING ${TOTAL_TIME} seconds" | tee -a ${result_path}
    fi
done

echo "SATISFIABLE: $SAT_COUNT" | tee -a ${result_path}
echo "UNSATISFIABLE: $UNSAT_COUNT" | tee -a ${result_path}
echo "time: -$TOTAL_TIME" | tee -a ${result_path}
