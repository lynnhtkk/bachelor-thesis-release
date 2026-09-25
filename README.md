# Delineating Substitutes and Complements in Grocery Data

Code release for the Bachelor's thesis *"Delineating Substitutes and
Complements in Grocery Data: A Semantically Guided Graph Traversal
Approach"* (Nyi Nyi Linn Htet, TUM School of Computation, Information and
Technology - Informatics; Examiner: Prof. Dr. Stephen Kobourov; Supervisor:
Dr. Jacob Miller).

The thesis detects grocery substitutes by combining product text with
transaction structure: products are embedded from their names, aisles, and
departments using a pretrained sentence encoder, then the bipartite
order-product graph is traversed with a random walk whose every candidate is
scored against the walk's fixed anchor (not its current position), so a
chain of individually plausible steps can't drift away from the anchor
product. The resulting sequences are filtered by a per-anchor similarity
threshold and used to train a Skip-gram model, and substitutes are retrieved
as nearest neighbors in the trained space. Six threshold configurations are
compared on the full Instacart dataset, judged by two independent LLMs on a
department- and frequency-stratified sample of anchor products.

This folder is a read-only, code-only copy for reference - it contains no
data, no trained models, and no run outputs, and is not meant to be executed
as-is. It exists so the thesis's methodology can be read alongside its text
without setting up the full pipeline environment.

## Dataset

All notebooks and scripts operate on the
[Instacart Market Basket Analysis dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
(orders, products, aisles, departments). Download it from Kaggle separately
if you want to reproduce the pipeline; it is not included here.
