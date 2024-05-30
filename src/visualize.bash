#!/usr/bin/env bash

DIR=$(dirname "$0")
DIR=${DIR%/}
CONFIG=
DATASET=
SKIP=false

usage() {
    echo \
    """
    Usage: bash visualize.bash 
                [--config <config>]
                [-d|--dataset <dataset>]
                [--skip-fn]
                [-h|--help]

    Options:
        --config <config>   Config file to use
        -d|--dataset <dataset>  Dataset to use (MOT, BDD)
        --skip-fn           Skip frames with only FN errors
    """
}

#? if no arguments are provided, return usage
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [ "$1" != "" ]; do
    case $1 in
        --config)
            CONFIG=$2
            shift
            shift
            ;;
        -d|--dataset)
            DATASET=$2
            shift
            shift
            ;;
        --skip-fn)
            SKIP=true
            shift
            ;;
        -h|--help)
            usage
            exit 1
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

#? Get output directory
IFS='/' read -ra EXP <<< "$CONFIG"
EXP=${EXP[-1]}
EXP=${EXP%.*}
EXP_DIR="/home/results/$EXP"

if [ ! -f "$DIR/$CONFIG" ]; then
    echo "Config file $CONFIG not found."
    exit 1
else
    echo "CONFIG... $CONFIG"
fi

RESULTS_FILE=$EXP_DIR/eval/results.pkl
OUT_DIR=$EXP_DIR/show_errors

ARGS="--result-file $RESULTS_FILE \
--out-dir $OUT_DIR"

#? Skip FN frames if --skip-fn is provided
if [ "$SKIP" = true ]; then
    echo "Skipping frames with FN errors only..."
    ARGS+=" --skip-fn"
fi

#? remove previous visualizations
[ -d "$OUT_DIR" ] && echo "Removing previous visualizations..." && \
rm -rf $OUT_DIR

cd $DIR
if [ "$DATASET" = "MOT" ]; then
    echo "Visualizing MOT17/20 results..."
    python -m mot_visualize $CONFIG \
        $ARGS
elif [ "$DATASET" = "BDD" ]; then
    echo "Visualizing BDD MOT results..."
    python -m bdd_visualize $CONFIG \
        $ARGS
else
    echo "Dataset not specified or not supported."
    exit 1
fi
