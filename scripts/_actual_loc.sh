cloc --include-ext=py \
bin/ \
cli.py \
model_discovery/evolution.py \
model_discovery/system.py \
model_discovery/utils.py \
model_discovery/ve/data_loader.py \
model_discovery/ve/evaluator.py \
model_discovery/ve/run.py \
model_discovery/model/composer.py \
model_discovery/model/gam.py \
model_discovery/model/loader.py \
model_discovery/model/block_registry.py \
model_discovery/model/utils/modules.py \
model_discovery/agents/flow/alang.py \
model_discovery/agents/flow/gau_flows.py \
model_discovery/agents/flow/gau_utils.py \
model_discovery/agents/roles/checker.py \
model_discovery/agents/roles/selector.py \
model_discovery/agents/roles/designer.py \
model_discovery/agents/agent_utils.py \
model_discovery/agents/search_utils.py \
model_discovery/agents/prompts/prompts.py \
model_discovery/configs/gam_config.py \
model_discovery/configs/const.py \


# Count only the core code (the code that will be used within the evolution)
# exclude similar code (e.g. claude.py) and unused code (e.g. reviewer.py)
# may include little non-core code (e.g. some parts from tester.py)
# there is actually some code from customized lm-eval, not counted here
# some codes are adapted elsewhere, they are not counted, e.g. trainer.py, etc., only few are counted, e.g. modules.py