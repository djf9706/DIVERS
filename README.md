# DIVERS
This is a pytorch implementation of the paper 

## Environment Requirement
The code has been tested running under Python 3.8.10 The required packages are as follows:
* pytorch == 1.13.0
* numpy == 1.22.3
* pandas == 1.4.2
* ray[tune] == 2.4.0
* bottleneck == 1.3.7
* protobuf == 3.19.0

## Dataset
* [Coat](https://www.cs.cornell.edu/~schnabts/mnar/)
* [Yahoo! R3](https://webscope.sandbox.yahoo.com/)
* [KuaiRand](https://kuairand.com/)
* Synthetic

## How to run the code
Take Coat as an example
1. Build the dataset via `dataset.py`
2. Build the embedding via `ivdp.py`
3. Train the ivae model to learn the latent confounder (add ``--tune `` for searching hyperparameters)

    ``python3 DIVERS_exposure.py  --dataset coat  --patience 100 ``
4. Save confounder models via `DIVERS_save_ae_params.py`
5. Run the feedback prediction model (add ``--tune `` for searching hyperparameters):

    ``
    python3 DIVERS.py --topk 5  --dataset coat --patience 20 
    ``
