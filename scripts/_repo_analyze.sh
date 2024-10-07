cloc --include-ext=py \
bin/ \
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
model_discovery/agents/flow/alang.py \
model_discovery/agents/flow/gau_flows.py \
model_discovery/agents/flow/gau_utils.py \
model_discovery/agents/roles/checker.py \
model_discovery/agents/roles/selector.py \
model_discovery/agents/roles/designer.py \
model_discovery/agents/agent_utils.py \
model_discovery/agents/search_utils.py \
model_discovery/agents/prompts/prompts.py \
model_discovery/model/library/tester.py 

# Count only the core code (the code that will be used within the evolution)
# exclude essentially same code (e.g. claude.py) 
# exclude code that may involves code from framework or other repo (e.g. modis_trainer.py, modules.py, the whole custom lm-eval)
# may include little non-core code (e.g. some parts from tester.py)