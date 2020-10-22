# KG-Conv-Rec

"Suggest me a movie for tonight": Leveraging Knowledge Graphs for Conversational Recommendation

<!-- ### [arXiv](https://arxiv.org/abs/1908.05391) -->

[Rajdeep Sarkar](https://dsi.nuigalway.ie/people/bb4f17b35edea725a26cf3e4e62bdd229d6ab978), [Koustava Goswami](https://dsi.nuigalway.ie/people/3c152032cea339067a0d1cb991e817271a98cf2c), [Mihael Arcan](https://dsi.nuigalway.ie/people/88b498677b9bf3780bf3d5151abbc5d69112dff8), [
John McCrae](https://dsi.nuigalway.ie/people/25cb2c4c1429e8dff2bdaa3a11fb8a5666d3872c).<br>
In The 28th International Conference on Computational Linguistics (COLING 2020)

## Prerequisites

- Python==3.6
- torch==1.3.0
- torch-cluster==1.4.5
- torch-geometric==1.3.2
- torch-sparse==0.4.3
- torchtext==0.6.0
- torchvision==0.4.1
## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/rajbsk/KG-conv-rec.git
cd KG-conv-rec/
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

- We use the **ReDial** dataset, which will be automatically downloaded by the script.
- The models and the dataset used in this work are stored in google drive. Download data.zip from [Google Drive]() and extract inside the KG-conv-rec folder.

### Training

To train the recommender part, run:

```bash
bash scripts/both.sh <subgraph_model_name> <num_exps> <gpu_id>
```
Where subgraph_model_name takes values 2_hop, 3_hop, 5_hop, pr, pr_0.7, pr_0.9 for the models build using subgraphs constructed using 2 hop, 3 hop, 5hop, PageRank, Personalized PageRank(alpha=0.7), Personalized PageRank(alpha=0.9) respectively. num_exps is the number of experiments and gpu_id is the id pf the gpu you want to run your code on.

### Logging

TensorBoard logs and models will be saved in `saved/` folder.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{sarkar2020kgrec,
  title={"Suggest me a movie for tonight": Leveraging Knowledge Graphs for Conversational Recommendation},
  author={Rajdeep Sarkar and Koustava Goswami and Mihael Arcan and John McCrae},
  booktitle = {Proceedings of the 28th International Conference on Computational
               Linguistics, {COLING} 2020, Barcelona, Spain, December 8-11,
               2020},
  year={2020}
}
```
