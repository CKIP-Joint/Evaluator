# MediaTek Research Traditional Chinese evaluation suite
This evaluation suite is intended to measure the knowledge and skill in Traditional Chinese of AI models. There are two completely new tests, TTQA and TCIC, proposed by MediaTek Research. In addition, there is one test translated from Simplified Chinese, TCWSC, and three existing Traditional Chinese tests. For ease of use, we follow the convention of [HELM](https://github.com/stanford-crfm/helm) to package these evaluation datasets into runnable evaluation routines.  

The **`crfm-helm`** Python package contains code used in the **Holistic Evaluation of Language Models** project ([paper](https://arxiv.org/abs/2211.09110), [website](https://crfm.stanford.edu/helm/v1.0/)) by [Stanford CRFM](https://crfm.stanford.edu/). 

To get started with **`crfm-helm`**, refer to [the documentation on Read the Docs](https://crfm-helm.readthedocs.io/) for how to install and run the package.

## Evaluation datasets for Traditional Chinese
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

## How to Set HTTP Port of HTTP client:
Modify `INFERENCE_ENDPOINT` in `src/helm/proxy/clients/huggingface_http_client.py`

## How to Use Three HF clients:
With following argument `--hf_mode`
1. `--hf_mode=hf`: `huggingface_client.py`: HF API
2. Not specified, `--hf_mode=tbi`: `huggingface_client_bloom_inference`: directly imoprt `transformers_bloom_inference`
3. `--hf_mode=http`:`huggingface_http_client`: use http request to query server hosted by `transformers_bloom_inference`

Three modules are defined in `src/helm/proxy/clients/auto_client.py`

## How to Add Model
1. Huggingface: 
    1. Create `_window_service.py` in `src/helm/benchmark/window_services`
    2. Modify `get_window_service` in `src/helm/benchmark/window_services/window_service_factory.py`
    3. Modify `ALL_MODEL` in `src/helm/proxy/models.py`
    4. Modify `src/helm/proxy/clients/huggingface_tokenizer.py` (Tokenizer is specified by `_window_service.py`)
    5. Modify `get_model_server_instance` in `src/helm/proxy/clients/huggingface_client_bloom_inference.py, huggingface_client.py, huggingface_http_client.py`

## How to Add Scenerio
1. Construct `<name>_scenerio.py`
    1. There should be at least a member function `get_instances(self) -> List[Instance]`
    2. Example of Instance:
        ```
        Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=split,
            )
        prompt: str
        answer: str
        ```
    3. If you need any local documents, put them in `restricted/<name>/` and read them in your `<name>_scenerio.py`.
2. Add `<name>_scenerio.py` into `src/helm/benchmark/scenarios`
3. import it in  `src/helm/benchmark/__init__.py`
4. Construct `get_<name>_spec` in `src/helm/benchmark/run_specs.py`. (Add it to `CANONICAL_RUN_SPEC_FUNCS`)
**Note** More details can be found in crfm-helm.readthedocs.io/en/latest/code/


## How to Summarize
`helm-summarize --output-path=? --suite v1 --skip-write-run-display-json`

## How to Evaluate a Lot of Checkpoint in a Folder.
You can use `evaluate_ckpt_in_folder.py`. **Read and modfify it carefully** before you run it.
The current logic of this script is
1. Split all tasks into several task sets. Each sets can be put on one cuda device.
2. ChangLe is not good at handling task dispatching, so after you modify the file, you need to run as following:
    - Assume you use cuda 0~3, you split the tasks to four sets, then please run
    `python evaluate_ckpt_in_folder.py 0`, `python evaluate_ckpt_in_folder.py 1`, `python evaluate_ckpt_in_folder.py 2`, `python evaluate_ckpt_in_folder.py 3` respectively.         
3. Once done, you can use summarizing tool to summarize.

## How to obtain summarizes .CSV Sheet after helm-summarize
Use `tools/summarized_json_to_df.py` and `tools/read_csvs_to_find_which_ckpt_is_best.py`.
**Read and modfify them carefully** before you run them.


