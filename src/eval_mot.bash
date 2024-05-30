#!/usr/bin/env bash

DIR=$(dirname "$0")
DIR=${DIR%/}
NUM_GPUS=1
CONFIG=
REMOVE=false
SHOW=false
EVAL=false

usage() {
    echo \
    """
    Usage: bash eval_mot.bash 
                [--gpus <num_gpus>]
                [--config <config>]
                [--show]
                [--eval]
                [--rm]

    Options:
        --gpus <num_gpus>   Number of GPUs to use
        --config <config>   Config file to use
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
        --gpus)
            NUM_GPUS=$2
            shift
            shift
            ;;
        --config)
            CONFIG=$2
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

if [ ! -f "$DIR/$CONFIG" ]; then
    echo "Config file $CONFIG not found."
    exit 1
else
    echo "CONFIG... $CONFIG"
fi

EVAL_DIR=$EXP_DIR/eval

ARGS="--work-dir $EVAL_DIR \
--out $EVAL_DIR/results.pkl"

#? Evaluate if --eval is provided
if [ "$EVAL" = true ]; then
    ARGS+=" --eval track"
fi

#? remove previous results
if [ "$REMOVE" = true ]; then
    echo "Replacing previous results..."
    [ -f "$EVAL_DIR/results.pkl" ] && rm $EVAL_DIR/results.pkl
fi

cd $DIR
if [[ $NUM_GPUS -gt 1 ]]
then
    #* NCCL_ASYNC_ERROR_HANDLING is enabled to use timeout arg for `init_process_group`
    #* see https://pytorch.org/docs/stable/distributed.html#initialization
    NCCL_ASYNC_ERROR_HANDLING=1 bash ./dist_test.sh $CONFIG $NUM_GPUS $ARGS
else
    #? Show results if --show is provided
    if [ "$SHOW" = true ]; then
        SHOW_DIR=$EXP_DIR/show
        [ -d "$SHOW_DIR" ] && echo "Removing previous visualizations..." && \
        rm -rf $SHOW_DIR
        mkdir -p $SHOW_DIR
        ARGS+=" --show-dir $SHOW_DIR"
    fi
    python -m test $CONFIG \
    $ARGS \
    --gpu-id 0
fi