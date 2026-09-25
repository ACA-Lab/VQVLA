# export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN to a Hugging Face access token before running this script}
rm /home/haozhe.jiang/Isaac-GR00T/logs/libero-goal.xlsx
cp /home/haozhe.jiang/Isaac-GR00T/logs/base.xlsx /home/haozhe.jiang/Isaac-GR00T/logs/libero-goal.xlsx

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/open_the_middle_drawer_of_the_cabinet \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_bowl_on_the_stove \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_wine_bottle_on_top_of_the_cabinet \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/open_the_top_drawer_and_put_the_bowl_inside \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_bowl_on_top_of_the_cabinet \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/push_the_plate_to_the_front_of_the_stove \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_cream_cheese_in_the_bowl \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/turn_on_the_stove \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_bowl_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_wine_bottle_on_the_rack \
    --n-action-steps 8 \
    --n-envs 1