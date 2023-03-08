# Traditional Chinese Evaluation Suite
This evaluation suite is intended to measure the knowledge and skills in Traditional Chinese of Language Models. For this purpose, MediaTek Research introduced two completely new evaluation tasks (TTQA and TCIC). In addition, this evaluation suite includes three existing Traditional Chinese evaluation tasks (DRCD, FGC, FLUD). For ease of use, we follow the convention of [HELM](https://github.com/stanford-crfm/helm) to package these evaluation datasets into runnable evaluation routines.  

The **`crfm-helm`** Python package contains code used originaly in the **Holistic Evaluation of Language Models** project ([paper](https://arxiv.org/abs/2211.09110), [website](https://crfm.stanford.edu/helm/v1.0/)) by [Stanford CRFM](https://crfm.stanford.edu/). 

To get started with **`crfm-helm`**, refer to [the documentation on Read the Docs](https://crfm-helm.readthedocs.io/) for how to install and run the package.

## Included Evaluation Datasets for Traditional Chinese
- TTQA: Taiwan Trivia Question Answering. Please refer to [paper]() 
- TCIC: Traditional Chinese Idiom Cloze. Please refer to [paper]()
- DRCD: Please refer to [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
- FGC: Please refer to [科技大擂台](https://scidm.nchc.org.tw/dataset/grandchallenge2020/resource/af730fe7-7f95-4af2-b4f4-1ca09406b35a)
- FLUD: Please refer to [科技大擂台](https://scidm.nchc.org.tw/dataset/grandchallenge2020)

## How to Run
See README below or:
1. Modify `run_specs.conf`
    - Note: `model=`  must be the last specified.
    - Note: For multiple choice, you need to specify method in .conf:
        1. `method=multiple_choice_joint`
        2. `method=multiple_choice_separate_original`
2. Run `python3 main.py --local --suite v1 -m=10 -c=run_specs.conf --num-thread=1`
    - Note: when using `huggingface_client_bloom_inference`, it is more stable to use `--num-thread=1`
    - Note: -m=max-eval-instances
    - When `huggingface/to_be_evaluated` is used in `run_specs.conf`, you can further use
        1. `--model_id=str` to specify a ID to change the subfolder name `huggingface/to_be_evaluated` in `--output-path`
        2. `--load_path` to specify ckpt path. If not specified, it will check `$EVALUATED_MODEL_PATH`
    - Note: use `--hf_batch_size=N` to inference with batchsize $N$

