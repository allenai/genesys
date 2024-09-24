

DEFAULT_TOKENIZER = 'meta-llama/Llama-2-7b-hf'
DEFAULT_CONTEXT_LENGTH = 2048

# GLUE_TASK_LIST = ["cola","mnli","mrpc","qnli","qqp","rte","sst","wnli"]

STANDARD_EVAL_TASKS = ["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande", "openbookqa"] 
ADDITIONAL_EVAL_TASKS = ["mathqa","sciq","swag","wsc273","scrolls_contractnli","scrolls_quality","qa4mre","triviaqa"]
BABYLM_GROUP = ["blimp_filtered","blimp_supplement"]
TINY_NON_STANDARD = ["tinyGSM8k", "tinyMMLU", "tinyTruthfulQA"]

GROUP_EVAL_TASKS = ["glue","inverse_scaling_mc","super-glue-lm-eval-v1","tinyBenchmarks","blimp"]
GENERATIVE_EVAL_TASKS = ["triviaqa", "drop", "babi", "unscramble", "scrolls_govreport","scrolls_summscreenfd","scrolls_narrativeqa","scrolls_qasper","scrolls_qmsum","squad_completion"]
DIFFICULT_EVAL_TASK = ["paws_en","race", "crows_pairs_english","commonsense_qa","anli","logiqa","headqa_en","hendrycks_math","chain_of_thought","gpqa_main_cot_zeroshot","ifeval","mmlu"]
NONACC_EVAL_TASKS = ["lambada_openai","cola","tinyGSM8k"]

# BUGGY TASKS: "storycloze_2016", "coqa"

# TODO: making a faster version of tasks that are MC only

# Other need generate: squad_completion

SMOLLM_125_CORPUS=['cosmopedia-v2','python-edu','fineweb-edu-dedup','open-web-math','deepmind-math-small','stackoverflow-clean']

# SMOLLM_125_CORPUS=['fineweb-edu-dedup']

DEFAULT_TASK_LIST1=[
    "inverse_scaling_mc",
    "glue",
    *STANDARD_EVAL_TASKS,
    *ADDITIONAL_EVAL_TASKS,
    *BABYLM_GROUP,
    *TINY_NON_STANDARD,
]


def MC_only(tasks):
    for t in GENERATIVE_EVAL_TASKS:
        if t in tasks:
            tasks.remove(t)
    for t in NONACC_EVAL_TASKS:
        if t in tasks:
            tasks.remove(t)
    return tasks