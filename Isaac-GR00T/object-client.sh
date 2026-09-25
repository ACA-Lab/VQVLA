# export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN to a Hugging Face access token before running this script}
rm /home/haozhe.jiang/Isaac-GR00T/logs/libero-object.xlsx
cp /home/haozhe.jiang/Isaac-GR00T/logs/base.xlsx /home/haozhe.jiang/Isaac-GR00T/logs/libero-object.xlsx

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_butter_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_milk_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket \
    --n-action-steps 8 \
    --n-envs 1