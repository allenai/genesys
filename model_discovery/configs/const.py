

DEFAULT_TOKENIZER = 'meta-llama/Llama-2-7b-hf'
DEFAULT_CONTEXT_LENGTH = 2048

STANDARD_EVAL_TASKS = ["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande", "openbookqa"] 
ADDITIONAL_EVAL_TASKS = ["squad_completion","mathqa","sciq","swag","wsc273","scrolls_contractnli","scrolls_quality","qa4mre"]
BABYLM_GROUP = ["blimp_filtered","blimp_supplement"]
TINY_NON_STANDARD = ["tinyGSM8k", "tinyMMLU", "tinyTruthfulQA"]

GROUP_EVAL_TASKS = ["glue","inverse_scaling_mc","super-glue-lm-eval-v1","tinyBenchmarks","blimp"]
GENERATIVE_EVAL_TASKS = ["triviaqa", "drop", "babi", "unscramble", "scrolls_govreport","scrolls_summscreenfd","scrolls_narrativeqa","scrolls_qasper","scrolls_qmsum"]
DIFFICULT_EVAL_TASK = ["paws_en","race", "crows_pairs_english","commonsense_qa","anli","logiqa","headqa_en","hendrycks_math","chain_of_thought","gpqa_main_cot_zeroshot","ifeval","mmlu"]


# BUGGY TASKS: "storycloze_2016", "coqa"

# TODO: making a faster version of tasks that are MC only
