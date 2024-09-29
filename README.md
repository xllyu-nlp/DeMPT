# DeMPT
This is the implementation for paper [DeMPT: Decoding-enhanced Multi-phase Prompt Tuning for Making LLMs Be Better Context-aware Translators](https://arxiv.org/abs/2402.15200). Unlike the concatenating strategy, this work splits the context-aware translation procedure into three phases to enhance the performance of context-aware machine translation:
<div align=center>

  ![msp](https://github.com/Rooders/DeMPT/blob/main/intro.png)

</div>

## Install Running Environment
We provide all of the dependencies in ``envs.yaml`` file. You can easily re-produce the running environment using ``conda`` via the following command:
```
conda env create -f envs.yml
```
Please refer to [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?tdsourcetag=s_pctim_aiomsg#viewing-a-list-of-the-packages-in-an-environment) for creating an environment from an ``*.yml`` file using ``conda``.

## Data Preparing
We provide some data examples in ``./data_exmaple``. Notably, you need to manually process the raw data in ``./raw_data`` into the data format in ``./data_exmaple``. ``<d>`` is used for marking the document boundary in ``./raw_data`` (original data can be downloaded from [here](https://data.statmt.org/news-commentary/v18)), such as:
```
doc_1_sent_1
doc_1_sent_2
<d>
doc_2_sent_1
doc_2_sent_2
```
Each line in the processed data includes a training instance, and each training instance includes three parts: 

```
[inter-sentence context] <extra_id_0> [source sentence] <extra_id_0> [target sentence]
```
``<extra_id_0>`` is the special token for the boundary of each part. ``[inter-sentence context]`` is consisted of the previous sentences of ``[source sentence]``. We set the length of ``inter-sentence context``to no more than 256.

## Training and Inference
We provide the training and inference pipeline in ``./exp_sh``. so you can reproduce our experimental results by running the corresponding pipeline, such as running:
```
bash ./exp_sh/exp-dempt-training.sh en zh  llama-2-7b-hf llama
```
to reproduce the DeMPT results in ZH-EN upon ``llama-2-7b``.

Before training and inference, please ensure the datasets have been processed into the directory ``./data_example`` or your customized path (need to change the ``data_dir`` in the corresponding shell script). Meanwhile, downloading the corresponding pretrained models, such as ``llama-2-7b-hf``
into ``./llms_ckpt``. The default saving directory is ``./workspace/$MODEL_TYPE``.

## Citation
```
@misc{lyu2024demptdecodingenhancedmultiphaseprompt,
      title={DeMPT: Decoding-enhanced Multi-phase Prompt Tuning for Making LLMs Be Better Context-aware Translators}, 
      author={Xinglin Lyu and Junhui Li and Yanqing Zhao and Min Zhang and Daimeng Wei and Shimin Tao and Hao Yang and Min Zhang},
      year={2024},
      eprint={2402.15200},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.15200}, 
}
```











