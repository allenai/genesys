TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']

DEFAULT_TOKENIZER = 'meta-llama/Llama-2-7b-hf'
DEFAULT_CONTEXT_LENGTH = 2048


DEFAULT_TOKEN_MULT = 20
DEFAULT_OPTIM = "adamw_hf"
DEFAULT_WANDB_PROJECT = 'model_discovery'
DEFAULT_WANDB_ENTITY = 'aristo'
DEFAULT_RANDOM_SEED = 42

DEFAULT_SAVE_STEPS = 50
DEFAULT_LOG_STEPS = 5



# GLUE_TASK_LIST = ["cola","mnli","mrpc","qnli","qqp","rte","sst","wnli"]

STANDARD_EVAL_TASKS = ["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande", "openbookqa"] 
ADDITIONAL_EVAL_TASKS = ["mathqa","sciq","swag","wsc273","qa4mre","triviaqa"]
# "scrolls_contractnli","scrolls_quality", # error
BABYLM_GROUP = ["blimp_filtered","blimp_supplement"]
TINY_NON_STANDARD = ["tinyGSM8k", "tinyMMLU", "tinyTruthfulQA"]

# Optional tasks
GROUP_EVAL_TASKS = ["glue","inverse_scaling_mc","super-glue-lm-eval-v1","tinyBenchmarks","blimp"]
GENERATIVE_EVAL_TASKS = ["triviaqa", "drop", "babi", "unscramble", "scrolls_govreport","scrolls_summscreenfd","scrolls_narrativeqa","scrolls_qasper","scrolls_qmsum","squad_completion"]
DIFFICULT_EVAL_TASK = ["paws_en","race", "crows_pairs_english","commonsense_qa","anli","logiqa","headqa_en","hendrycks_math","chain_of_thought","gpqa_main_cot_zeroshot","ifeval","mmlu"]
NONACC_EVAL_TASKS = ["lambada_openai","cola","tinyGSM8k"]

# BUGGY TASKS: "storycloze_2016", "coqa"

# Other need generate: squad_completion

SMOLLM_125_CORPUS=['cosmopedia-v2','python-edu','fineweb-edu-dedup','open-web-math','deepmind-math-small','stackoverflow-clean']

DEFAULT_TASK_LIST1=[
    "inverse_scaling_mc",
    "glue",
    # "smollm125-tiny",  
    "squad_completion",
    *STANDARD_EVAL_TASKS,
    *ADDITIONAL_EVAL_TASKS,
    *BABYLM_GROUP,
    *TINY_NON_STANDARD,
]

DEFAULT_TRAINING_DATA=SMOLLM_125_CORPUS
DEFAULT_EVAL_TASKS=DEFAULT_TASK_LIST1


def MC_only(tasks):
    for t in GENERATIVE_EVAL_TASKS:
        if t in tasks:
            tasks.remove(t)
    for t in NONACC_EVAL_TASKS:
        if t in tasks:
            tasks.remove(t)
    return tasks