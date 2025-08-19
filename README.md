<div align="center">
  
  <div>
  <h1>ADMIRE-BayesOpt: Accelerated Data MIxture
RE-weighting for Language Models with Bayesian
Optimization</h1>
  </div>

  <!-- <div>
      Thomas Hartvigsen&emsp; Swami Sankaranarayanan&emsp; Hamid Palangi&emsp; Yoon Kim&emsp; Marzyeh Ghassemi
  </div> -->
  <!-- <br/> -->

</div>

Official implementation of **[ADMIRE-BayesOpt: Accelerated Data MIxture
RE-weighting for Language Models with Bayesian
Optimization](https://xo28.github.io/ADMIRE-BayesOpt-homepage/)**.
Please feel free to email us or raise an issue with this repository and we'll get back to you as soon as possible.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Setup
1. Create a virtual environment (we use conda)
2. Activate the virtual environment
3. Install the repository
    ```
    conda env create --name admire_bayesopt
    conda activate admire_bayesopt
    pip install -r requirements.txt
    ```
## 
This implementaion is based on an official BoTorch tutorial: [Multi-fidelity Bayesian optimization with discrete fidelities using KG](https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/). We followed its comparasions between BayesOpt and MFBayesOpt.

## Data Preparation
We opensource the data mixture dataset: ```admire_ift_runs``` and use the mixture dataset on the Pile ```regmix-data``` from [RegMix](https://github.com/sail-sg/regmix/tree/main).
We run experiments of different mixtures with [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) 0.5B / 3B / 7B.

## Running experiments
Choose the index of target domain: ```--idx```. 
<br/>
Choose the dataset [admire_ift_runs/pile]: ```--dataset```.
<br/>
Results will be saved in ```saved_logs```.

### Training and recommending with BayesOpt on admire_ift_runs.
```
python bayesopt_admire_ift_runs.py --idx -3  #average of ood+id
```

### Training and recommending with BayesOpt on the Pile
```
python bayesopt_thepile.py --idx -1 #average
```

### Training and recommending with MFBayesOpt on admire_ift_runs / the Pile
```
python mfbayesopt_maxvalue.py --dataset admire_ift_runs --idx -3 #average of ood+id
python mfbayesopt_maxvalue.py --dataset pile --idx -1 #average
```

## Citation
Please use the following to cite this work:
```
@misc{chen2025admirebayesoptaccelerateddatamixture,
      title={ADMIRE-BayesOpt: Accelerated Data MIxture RE-weighting for Language Models with Bayesian Optimization}, 
      author={Shengzhuang Chen and Xu Ouyang and Michael Arthur Leopold Pearce and Thomas Hartvigsen and Jonathan Richard Schwarz},
      year={2025},
      eprint={2508.11551},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2508.11551}, 
}
```
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
