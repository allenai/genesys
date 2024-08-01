import os,sys
import datasets
from pathlib import Path

from datasets import load_dataset
import shutil

mkdir=os.makedirs

data_dir='../data/fineweb_edu_temp'  
mkdir(data_dir, exist_ok=True)
datasets.config.DOWNLOADED_DATASETS_PATH = Path(data_dir)

export_dir='../data/fineweb_edu_4'


FWE_SUBSETS=[
    "CC-MAIN-2013-20",
    "CC-MAIN-2013-48",
    "CC-MAIN-2014-10",
    "CC-MAIN-2014-15",
    "CC-MAIN-2014-23",
    "CC-MAIN-2014-35",
    "CC-MAIN-2014-41",
    "CC-MAIN-2014-42",
    "CC-MAIN-2014-49",
    "CC-MAIN-2014-52",
    "CC-MAIN-2015-06",
    "CC-MAIN-2015-11",
    "CC-MAIN-2015-14",
    "CC-MAIN-2015-18",
    "CC-MAIN-2015-22",
    "CC-MAIN-2015-27",
    "CC-MAIN-2015-32",
    "CC-MAIN-2015-35",
    "CC-MAIN-2015-40",
    "CC-MAIN-2015-48",
    "CC-MAIN-2016-07",
    "CC-MAIN-2016-18",
    "CC-MAIN-2016-22",
    "CC-MAIN-2016-26",
    "CC-MAIN-2016-30",
    "CC-MAIN-2016-36",
    "CC-MAIN-2016-40",
    "CC-MAIN-2016-44",
    "CC-MAIN-2016-50",
    "CC-MAIN-2017-04",
    "CC-MAIN-2017-09",
    "CC-MAIN-2017-13",
    "CC-MAIN-2017-17",
    "CC-MAIN-2017-22",
    "CC-MAIN-2017-26",
    "CC-MAIN-2017-30",
    "CC-MAIN-2017-34",
    "CC-MAIN-2017-39",
    "CC-MAIN-2017-43",
    "CC-MAIN-2017-47",
    "CC-MAIN-2017-51",
    "CC-MAIN-2018-05",
    "CC-MAIN-2018-09",
    "CC-MAIN-2018-13",
    "CC-MAIN-2018-17",
    "CC-MAIN-2018-22",
    "CC-MAIN-2018-26",
    "CC-MAIN-2018-30",
    "CC-MAIN-2018-34",
    "CC-MAIN-2018-39",
    "CC-MAIN-2018-43",
    "CC-MAIN-2018-47",
    "CC-MAIN-2018-51",
    "CC-MAIN-2019-04",
    "CC-MAIN-2019-09",
    "CC-MAIN-2019-13",
    "CC-MAIN-2019-18",
    "CC-MAIN-2019-22",
    "CC-MAIN-2019-26",
    "CC-MAIN-2019-30",
    "CC-MAIN-2019-35",
    "CC-MAIN-2019-39",
    "CC-MAIN-2019-43",
    "CC-MAIN-2019-47",
    "CC-MAIN-2019-51",
    "CC-MAIN-2020-05",
    "CC-MAIN-2020-10",
    "CC-MAIN-2020-16",
    "CC-MAIN-2020-24",
    "CC-MAIN-2020-29",
    "CC-MAIN-2020-34",
    "CC-MAIN-2020-40",
    "CC-MAIN-2020-45",
    "CC-MAIN-2020-50",
    "CC-MAIN-2021-04",
    "CC-MAIN-2021-10",
    "CC-MAIN-2021-17",
    "CC-MAIN-2021-21",
    "CC-MAIN-2021-25",
    "CC-MAIN-2021-31",
    "CC-MAIN-2021-39",
    "CC-MAIN-2021-43",
    "CC-MAIN-2021-49",
    "CC-MAIN-2022-05",
    "CC-MAIN-2022-21",
    "CC-MAIN-2022-27",
    "CC-MAIN-2022-33",
    "CC-MAIN-2022-40",
    "CC-MAIN-2022-49",
    "CC-MAIN-2023-06",
    "CC-MAIN-2023-14",
    "CC-MAIN-2023-23",
    "CC-MAIN-2023-40",
    "CC-MAIN-2023-50",
    "CC-MAIN-2024-10",
]


WORD_SIZE=1 # multi proc actually not working because of network collisions
RANK=sys.argv[1] if len(sys.argv) >= 2 else 0
each=int(len(FWE_SUBSETS)/WORD_SIZE)
low=int(RANK)*each
high=(int(RANK)+1)*each
if int(RANK)==WORD_SIZE-1:
    high=len(FWE_SUBSETS)
print(f"Processing {low} to {high}, rank {RANK}, word size {WORD_SIZE}")

for idx,subset in enumerate(FWE_SUBSETS):
    if idx < low or idx >= high:
        continue
    export_dir_dataset = os.path.join(export_dir,subset)
    try:
        ss=load_dataset(export_dir_dataset)
        assert ss.num_rows['train'] > 0
        print(f"Already processed: {subset}")
        continue
    except:
        pass
    print(f"Processing {idx+1}/{len(FWE_SUBSETS)}: {subset}")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", subset)
    ds_filtered = ds.filter(lambda x: x['score'] >= 4.0, num_proc=28)
    ds_filtered.save_to_disk(export_dir_dataset) 
    print(f"Saved {subset}")
    if load_dataset(export_dir_dataset).num_rows['train'] == 0:
        print(f"Empty dataset: {subset}")
        shutil.rmtree(export_dir_dataset, ignore_errors=True)
    else:
        ds.cleanup_cache_files()

print("Done")
