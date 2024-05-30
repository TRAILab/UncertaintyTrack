#!/usr/bin/env bash

DIR=$(dirname "$0")
DIR=${DIR%/}
CONFIG=
CHECKPOINT=
REMOVE=false
SHOW=false
EVAL=false

usage() {
    echo \
    """
    Usage: bash eval_det.bash 
                [--config <config>]
                [--checkpoint <checkpoint path>]
                [--show]
                [--eval]
                [--rm]

    Options:
        --config <config>   Config file to use
        --checkpoint <checkpoint path>   Checkpoint to use for evaluation
        --show              Show visualization results
        --eval              Evaluate results with metrics
        --rm                Remove previous results
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
        --checkpoint)
            CHECKPOINT=$2
            shift
            shift
            ;;
        --show)
            SHOW=true
            shift
            ;;
        --eval)
            EVAL=true
            shift
            ;;
        --rm)
            REMOVE=true
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

if [ ! -f "$CONFIG" ]; then
    echo "Config file $CONFIG not found."
    exit 1
else
    echo "CONFIG... $CONFIG"
fi

EVAL_DIR=$EXP_DIR/eval

ARGS="--work-dir $EVAL_DIR \
--checkpoint $CHECKPOINT \
--out $EVAL_DIR/results.pkl"

#? Evaluate if --eval is provided
if [ "$EVAL" = true ]; then
    ARGS+=" --eval bbox scoring"
fi

#? Show results if --show is provided
if [ "$SHOW" = true ]; then
    ARGS+=" --show-dir $EVAL_DIR/show"
fi

#? remove previous results
if [ "$REMOVE" = true ]; then
    echo "Removing previous results..."
    [ -f "$EVAL_DIR/results.pkl" ] && rm $EVAL_DIR/results.pkl
fi

cd $DIR
python -m test $CONFIG \
    $ARGS \
    --gpu-id 0