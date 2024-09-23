# DeMPT
This is the implementation for paper DeMPT: Decoding-enhanced Multi-phase Prompt Tuning for Making LLMs Be Better Context-aware Translators. Unlike the concatenating strategy, this work splits the context-aware translation procedure into three phases to enhance the performance of context-aware machine translation:
<div align=center>

  ![msp](https://github.com/Rooders/DeMPT/blob/main/intro.png)

</div>

## Install Running Environment
We provide all of the dependencies in ``envs.yaml`` file. You can easily re-produce the running environment using ``conda`` via the following commands:
```
conda env create -f environment.yml
```
Please refer to [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?tdsourcetag=s_pctim_aiomsg#viewing-a-list-of-the-packages-in-an-environment) for creating an environment from an ``*.yml`` file using ``conda``.

## Data Prepare
We provide the data example in ``./data_exmaple``. Notably, you need to process the raw data in ``./raw_data`` into the data format in ``./data_exmaple``. Each line includes a training instance, and each training instance includes three part: 

```

```
