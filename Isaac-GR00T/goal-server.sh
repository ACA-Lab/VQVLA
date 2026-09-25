# export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN to a Hugging Face access token before running this script}

python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper