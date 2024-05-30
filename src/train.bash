#!/usr/bin/env bash

DIR=$(dirname "$0")
DIR=${DIR%/}
NUM_GPUS=
CONFIG=
REMOVE=false
RESUME=false
EVAL=true

usage() {
    echo \
    """
    Usage: bash train.bash 
                [--gpus <num_gpus>]
                [--config <config>]
                [--rm]
                [--no-eval]
                [--resume]

    Options:
        --gpus <num_gpus>   Number of GPUs to use
        --config <config>   Config file to use
        --rm                Remove previous results
        --no-eval           Turn off validation
        --resume            Resume training from same folder
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
        --rm)
            REMOVE=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-eval)
            EVAL=false
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
OUTPUT="/home/results/$EXP"

if [ ! -f "$DIR/$CONFIG" ]; then
    echo "Config file $CONFIG not found."
    exit 1
else
    echo "CONFIG... $CONFIG"
fi

if [ "$REMOVE" = true ]; then
    echo "Removing previous results..."
    [ -d $OUTPUT ] && rm -rf $OUTPUT
fi

echo "Saving results to... $OUTPUT"

ARGS="--work-dir $OUTPUT \
--seed 0"

if [ "$RESUME" = true ]; then
    echo "Resuming training..."
    [ -f "${OUTPUT}/latest.pth" ] && \
    ARGS+=" --resume-from ${OUTPUT}/latest.pth"
fi

if [ "$EVAL" = false ]; then
    echo "Training without validation..."
    ARGS+=" --no-validate"
fi

cd $DIR
if [[ $NUM_GPUS -gt 1 ]]
then
    #* NCCL_ASYNC_ERROR_HANDLING is enabled to use timeout arg for `init_process_group`
    #* see https://pytorch.org/docs/stable/distributed.html#initialization
    NCCL_ASYNC_ERROR_HANDLING=1 bash ./dist_train.sh $CONFIG $NUM_GPUS $ARGS
else
    python -m train $CONFIG \
    $ARGS \
    --gpu-id 0
fi