export DEVICES="0,1,4,5"

deepspeed --include localhost:${DEVICES} ./ds_baseline.py --deepspeed_config=./ds_config_gpt_j.json
# deepspeed --include localhost:${DEVICES} ./hf_baseline.py --deepspeed_config=./ds_config_gpt_j.json