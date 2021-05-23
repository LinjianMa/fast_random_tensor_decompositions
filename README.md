## Fast randomized tensor decompositions
This library contains the implementation of the paper "Fast and accurate randomized algorithms for low-rank tensor decompositions".

### Note:

Run
```
pip install -r requirements.txt
```
to install necessary packages. 

Run 

```
python run_als.py -h
```
to see the existing input arguments and their functions.

To run Tucker decomposition on the dense random tensor (tensor 1 or tensor 2 in the paper), run
```
python run_als.py --s 500 --R 5 --epsilon 0.25 --seed 1 --tensor random --hosvd 3 --decomposition Tucker --rank-ratio 1.2 --fix-percentage 0. --num-iter 10 --method Leverage
```
The input tensor will have size `s x s x s`, Tucker rank is based on `--hosvd-core-dim`, the sketch size is `(R/epsilon)**2`. `--hosvd 0` means random initialization, 1 means initialize with HOSVD, and 3 means initialize with RRF detailed in the paper.

To run tensor 2 in the paper, set `--tensor random_bias`. 

Method can be `ALS`, means the traditional ALS algorithm, or `Leverage`, meaning leverage score sampling (when setting `--fix-percentage 0` it's random sampling, when setting `--fix-percentage 1` it's deterministic sampling), or `Tensorsketch`, meaning using the TensorSketch algorithm, or `Tensorsketch-ref`, meaning running the algorithm proposed in Melik and Becker, NeurIPS 2018.

To run Tucker decomposition on the real image dataset, run
```
python run_als.py --R 5 --epsilon 0.25 --seed 1 --tensor coil100 --hosvd 3 --decomposition Tucker --fix-percentage 0. --num-iter 10 --method Leverage
```
where `--tensor` can also be `timelapse`.

To run Tucker decomposition on sparse random tensors (detailed in the appendix), run
```
python run_als.py --s 500 --R 5 --epsilon 0.25 --seed 1 --tensor random --hosvd 3 --decomposition Tucker_simulate --rank-ratio 1.2 --fix-percentage 0. --num-iter 10 --method Leverage --sparsity 0.2
```
where `--sparsity` denotes the `p` parameter in the paper.

To run CP decomposition on sparse random tensors, run
```
python run_als.py --s 500 --R 5 --epsilon 0.25 --seed 1 --tensor random --hosvd 3 --decomposition CP_simulate --rank-ratio 1.2 --fix-percentage 0. --num-iter 10 --method Leverage --sparsity 0.2
```
For this case, method can be `ALS`, meaning traditional ALS algorithm, or `Leverage`, meaning CP-ALS with leverage score sampling, or `Tucker`, meaning run the Tucker+CP algorithm, or `Leverage_tucker`, meaing run the leverage sampling based Tucker+CP algorithm.
