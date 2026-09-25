# export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN to a Hugging Face access token before running this script}
rm /home/haozhe.jiang/Isaac-GR00T/logs/libero-spatial.xlsx
cp /home/haozhe.jiang/Isaac-GR00T/logs/base.xlsx /home/haozhe.jiang/Isaac-GR00T/logs/libero-spatial.xlsx

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate \
    --n-action-steps 8 \
    --n-envs 1