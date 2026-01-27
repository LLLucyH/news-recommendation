# Candidate-Aware News Recommendation System

This repository implements a deep learning-based news recommendation system using the **Microsoft News Dataset (MIND)**. The project features a custom **Candidate-Aware User Encoder** that utilizes Multi-Head Cross-Attention to dynamically model user interests relative to specific news items.

## Problem Definition

Predicting user clicks on news is challenging because a user’s historical clicks (e.g., "Sports" and "Finance") might not all be relevant to a new article (e.g., a "Technology" update). Our model addresses this by providing a personalized recommendation score that focuses on the specific parts of a user's history that align with the candidate news article.

## Methodology

The model follows a specialized architecture designed for content-heavy environments:

1. **News Encoder**: Both the candidate news and history items pass through a shared encoder. It fuses:
* **Text Features**: Uses `DistilBERT` (distilbert-base-uncased) to encode news titles. We chose DistilBERT over the full BERT model to reduce the computational footprint during training.
* **Knowledge Graph Features**: Integrated by extracting Wikidata entity IDs from the news metadata. These are mapped to pre-trained vectors and merged with the LLM output via a projection MLP.

2. **Candidate-Aware User Encoder**:
* Employs **Multi-Head Cross-Attention**.
* The **Candidate News** vector acts as the **Query (Q)**.
* The **User History** vectors act as **Keys (K)** and **Values (V)**.
* This allows the model to attend more strongly to history items related to the current candidate.

3. **Prediction**: A sigmoid-activated inner product between the context-aware user vector and the candidate vector.

<br>
<img width="2000" height="1026" alt="architecture_diagram-1" src="https://github.com/user-attachments/assets/d83df18a-2a19-4231-8ac4-7ffa8fa42c99" />

**Figure 1: Schematic of the Candidate-Aware News Recommendation System.** The model employs a dual-encoder framework: a News Encoder that fuses semantic features from text (via pre-trained DistilBERT) with structural world knowledge (via TransE Graph Embeddings), and a Candidate-Aware User Encoder. The user encoder utilizes a Multi-Head Cross-Attention mechanism where the candidate news acts as the Query ($Q$) to dynamically weight and aggregate the user's historical clicked news (Keys $K$ and Values $V$), producing a context-specific user representation for final click probability scoring.

<br>

### Why Certain Design Choices Were Made
* **Small Set Selection**: Due to computational constraints, we conducted our primary experiments on the **MIND-small** version of the dataset.
* **Cross-Attention vs. Contrastive Loss**: We considered a contrastive loss between user and candidate vectors but decided against it because cross-attention already explicitly models the interaction between the user and candidate, and contrastive learning typically requires very large batch sizes to be effective, which was not feasible under our hardware constraints.
* **Title-Only Focus**: We prioritized news titles as they offer a high signal-to-noise ratio, allowing for faster training iterations compared to using long abstracts.
* **Hyperparameter optimization**: We optimized eight hyperparameters, which would have been computationally expensive in a grid-search setup. We therefore used random search over ten different configurations.

### Knowledge Graph Embedding Calculation
The graph embeddings are 100-dimensional vectors provided by the MIND dataset, calculated using the TransE (Translating Embeddings) method.
* **Calculation Logic**: TransE models relationships as translations in the vector space. For a triplet $(head, relation, tail)$, it optimizes the embedding such that $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$.
* **Training**: The model is learned by distinguishing positive triplets from corrupted (negative) ones using a margin-based ranking loss.
* **Integration**: We map Wikidata IDs in our news text to these pre-computed vectors to provide structural "world knowledge" context that text alone may lack.

### Baseline Models

Two traditional machine learning baselines are provided for comparison (implemented in `train_baseline.py`):

* **TF-IDF + SVD**: Applies Latent Semantic Analysis (LSA) to news titles to create dense feature vectors.
* **Entity Embeddings**: Uses only the pre-trained knowledge graph embeddings.
* *Note: Baselines represent the user by mean-pooling their history vectors.*

## Results
The following figure summarizes performance across multiple metrics:
<img width="2867" height="1673" alt="results-1" src="https://github.com/user-attachments/assets/0482ceb9-0227-4c02-b048-9bfc93ebcd4e" />
**Figure 2: Performance Evaluation across Baseline and Proposed Configurations.** The bar charts compare the model performance on the MIND-small dataset using standard ranking metrics: AUC, nDCG@5, nDCG@10, and MRR. The results demonstrate the incremental gains provided by pre-trained LLM features and knowledge graph integration. Error bars represent bootstrapped 95% confidence intervals.
<br><br>

### Results Analysis and Ablations
The results demonstrate several key insights:

* **Impact of Transfer Learning**: The significant performance jump in the pretrained versions shows that the LLM (DistilBERT) captures semantic nuances in news titles that standard word embeddings miss.

* **Impact of Graph Embeddings**: Models utilizing knowledge graph features consistently show higher `nDCG@10` and `MRR`. This indicates that structural knowledge helps bridge the gap when words in a title do not directly overlap. Nonetheless, this effect is less impactful than the impact of transfer learning with pretrained LLMs.

* **Ablation Success**: The `Pretrained + Graph` version is the strongest configuration. The synergy between high-level linguistic features and structural entity data provides the most robust representation for ranking.

### Possible Extensions
Future work could explore hard negative mining to improve discriminative power by selecting unclicked news that is content-similar to the history. Incorporating temporal decay would allow the attention mechanism to prioritize more recent clicks. Finally, scaling to the MIND-large dataset with distributed training could help the model generalize to rarer topics.

## Usage Instructions

### Environment Setup

Ensure you have Python 3.8+ and the required packages installed:

```bash
pip install -r requirements.txt

```

### Downloading the Data
The MIND data can be downloaded from the [GitHub](https://msnews.github.io/).

### Downloading the DistilBERT Model

The DistilBERT weights are approximately **260MB**, which exceeds GitHub's standard 100MB file limit.

* **Automatic Download**: The code is configured to fetch the model automatically from Hugging Face via the `transformers` library on the first run.
* **Manual Setup**: If you are on a restricted network, download the `distilbert-base-uncased` files manually, place them in a local directory, and update the `MODEL_NAME` in `src/config.py`.

### Running Training

To train the main model (supports Multi-GPU/DDP):

```bash
python train.py --data_root ./path/to/data --gpus 2 --batch_size 64

```

To train the baselines:

```bash
# TF-IDF Baseline
python train_baseline.py --method tfidf --svd_dim 100

# Entity Embedding Baseline
python train_baseline.py --method entity

```

To run hyperparameter optimization with random search:

```bash
python random_search.py

```
