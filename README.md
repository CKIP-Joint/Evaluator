# MediaTek Research Traditional Chinese evaluation suite
This evaluation suite is intended to measure the knowledge and skill in Traditional Chinese of AI models. There are two completely new tests, TTQA and TCIC, proposed by MediaTek Research. In addition, there is one test translated from Simplified Chinese, TCWSC, and three existing Traditional Chinese tests. For ease of use, we follow the convention of [HELM](https://github.com/stanford-crfm/helm) to package these evaluation datasets into runnable evaluation routines.  

## Installation
1. clone [HELM](https://github.com/stanford-crfm/helm)
2. Put all files in `scenario` into `helm/src/helm/benchmark/scenarios`
3. Put `restricted` in the root of [HELM](https://github.com/stanford-crfm/helm). 
4. Following the [instruction](https://crfm-helm.readthedocs.io/en/latest/code/) to add our scenarios into the HELM.

## Introduction
- TTQA: Taiwan Triva Question Answering. Please refer to [paper]() 
- TCIC: Traditional Chinese Idiom Cloze. Please refer to [paper]()
- TCWSC: Traditional Chinese Winograd Schema Challenge. Please refer to [paper]() 
- DRCD: Please refer to [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
- FGC: Please refer to [科技大擂台](https://scidm.nchc.org.tw/dataset/grandchallenge2020/resource/af730fe7-7f95-4af2-b4f4-1ca09406b35a)
- FLUD: Please refer to [科技大擂台](https://scidm.nchc.org.tw/dataset/grandchallenge2020)
