# Theoretical Background

This document summarises the mathematical foundations used in the toolkit, following the notation from the *Network Data Analysis* course (M.F. Marino, University of Florence).

> **Note**: This theory chapter is extracted from the main README for clarity. For usage instructions and examples, see [README.md](README.md).

---

## Table of Contents

0. [Introduction: Statistical Modelling of Networks](#0-introduction-statistical-modelling-of-networks)
1. [Basic Notation](#basic-notation)
2. [Global Statistics: Density, Reciprocity, Transitivity](#1-global-statistics-density-reciprocity-transitivity)
3. [Node Centrality and Network Centralisation](#2-node-centrality-and-network-centralisation)
4. [Stochastic Block Model (SBM)](#3-stochastic-block-model-sbm)
5. [SBM vs Modularity-based Methods](#4-sbm-vs-modularity-based-methods)
6. [Assortativity Coefficient](#assortativity-coefficient)
7. [Temporal Extension (Dynamic SBM)](#5-temporal-extension-dynamic-sbm--project-extension)
8. [Hypergraph Group Extraction via Cliques](#6-hypergraph-group-extraction-via-cliques)

---

## 0. Introduction: Statistical Modelling of Networks

### Why Statistical Models for Networks?

Network data are characterised by a number of **dependencies** which have been found both empirically and theoretically:
- **Reciprocation**: if $i$ connects to $j$, does $j$ connect back to $i$?
- **Homophily**: do similar nodes tend to connect?
- **Transitivity**: are friends of friends also friends?
- **Degree variability**: are some nodes more connected than others?

Under a statistical modelling approach, the observed network is considered as *an outcome of a random draw from the postulated model*. It is natural to consider that the observed network data could have been different — different nodes, different timing, different external influences. However, in such a "population" of different networks, systematic patterns captured by the model parameters would remain the same.

The aim of the statistical model is to represent the main features of the network via a small number of parameters. Expressing the uncertainty of those estimates gives an indication of how different estimates might be if the researcher had observed a different network from the "population".

### Three Broad Approaches

The literature distinguishes three broad approaches to account for network dependencies:

| Approach | Description | Key Idea |
|----------|-------------|----------|
| **Incorporating network structure through covariates** | A statistical model for independent data is considered; network dependencies are represented by including them as explanatory variables | Used mainly in longitudinal settings where earlier observations produce covariates (Gulati & Gargiulo, 1999) |
| **Controlling for network dependencies** | Network dependencies are considered by specifying a covariance structure but are not explicitly modelled | E.g., Lindgren (2010) |
| **Modelling network structure** | Structural dependencies between tie variables $Y_{ij}$ are explicitly modelled; a potentially large number of parameters is included | The observed network is a random draw from the postulated model |

This toolkit focuses on the **third approach**: explicitly modelling network structure.

### Modelling Alternatives: A Historical Overview

Different statistical models have been developed in the literature to express network dependencies:

#### 1. Conditionally Uniform Models (Holland & Leinhardt, 1976)

- Network properties that researchers wish to control for are summarised by means of proper network statistics
- It is assumed that, conditional on these statistics, the distribution of the network is **uniform**
- Each network satisfying the constraints has the same probability; all others have probability zero
- Holland & Leinhardt considered conditioning statistics from the **dyad census** (mutual, asymmetric, null dyads)
- Snijders (1991) extended this to in- and out-degree distributions
- **Limitation**: These models become very complicated when more elaborate properties and richer conditioning statistics are considered; they are not widely used today

#### 2. Latent Space Models

These models assume the existence of **latent (unobserved) variables**, such that the observed variables have a simple probability distribution given the latent variables.

- The specification of the latent variables' distribution identifies the **structural model**
- The distribution of the observed variables, conditional on the latent ones, identifies the **measurement model**

Two main types:

| Type | Description | Key References |
|------|-------------|----------------|
| **Discrete Latent Space Models** | Node-level latent variables indicate block membership; conditional on them, tie variables are independent | Holland et al. (1983), Nowicki & Snijders (2001), Daudin et al. (2008), Airoldi et al. (2008) |
| **Distance Latent Space Models** | Nodes are projected in a low-dimensional latent space (2–3 dimensions); the probability of a tie decreases with distance in this space | Hoff et al. (2002) |

**Stochastic Block Models (SBMs)** are the most prominent example of discrete latent space models. The **Mixed Membership Model** (Airoldi et al., 2008) extends SBMs by allowing each node to belong to multiple clusters, describing situations where nodes play multiple roles.

#### 3. Exponential Random Graph Models (ERGMs)

- Rather than conditioning on latent variables, the dependence between tie variables is **explicitly modelled**
- The probability of observing the network is directly modelled as a function of given network statistics (sufficient statistics)
- These statistics are defined to capture dependencies between tie variables (e.g., triangles, two-stars)
- Key reference: Frank & Strauss (1986)

### Why This Toolkit Focuses on SBMs

This toolkit implements **Stochastic Block Models** as the primary method for community detection because:

1. **Principled inference**: SBMs provide a generative probabilistic framework with well-defined likelihood
2. **Model selection**: The number of blocks can be selected via principled criteria (ICL, MDL)
3. **Interpretability**: Block membership has clear interpretation as latent community structure
4. **Flexibility**: SBMs can detect assortative, disassortative, and core-periphery structures
5. **Uncertainty quantification**: Posterior probabilities provide uncertainty estimates for block assignments

---

## Basic Notation

We consider a network on $n$ nodes $\{1,\dots,n\}$ described by the random adjacency matrix $Y$ and its observed realisation $y$.
The element $Y_{ij}$ is the dyadic variable representing the relationship between nodes $i$ and $j$.  
In the binary case, $Y_{ij} = 1$ if there is an edge between $i$ and $j$, and $Y_{ij} = 0$ otherwise.

For undirected networks, symmetry holds: $Y_{ij}=Y_{ji}$; for directed networks, $Y_{ij}$ and $Y_{ji}$ may differ.

---

## 1. Global Statistics: Density, Reciprocity, Transitivity

### Density

The density $\rho$ is the ratio between observed ties and possible ties:

$$\rho = \frac{m}{m_{\max}}$$

where $m$ is the number of observed ties and $m_{\max}$ is the maximum possible number of ties.

For directed networks, the denominator is $n(n-1)$; for undirected networks, $n(n-1)/2$.
In practice, starting from $y$, the numerator is computed as $\sum_i\sum_{j\neq i} y_{ij}$ (directed) or $\sum_i\sum_{j<i} y_{ij}$ (undirected).
Moreover, $\rho \approx \Pr(Y_{ij}=1)$, i.e., it estimates the probability of observing a tie between two randomly selected nodes.

### Reciprocity (directed networks)

Reciprocity $R$ is the fraction of reciprocated ties:

$$R = \frac{\sum_{ij} y_{ij}y_{ji}}{m}$$

where $m$ is the number of observed ties.

> *Note: Reciprocity applies only to directed networks. The LyonSchool network analysed in this project is undirected, so reciprocity is not computed.*

### Transitivity / Clustering

For undirected networks, transitivity measures how much "friends of friends" tend to be friends.
One formulation is the coefficient:

$$C = \frac{\text{(closed paths of length 2)}}{\text{(paths of length 2)}}$$

Equivalently, via triad census, we count the number of triangles and connected triplets (two-stars). Then:

$$C = \frac{\text{(triangles)}}{\text{(triangles)} + \text{(two-stars)}}$$

where connected triplets = triangles + two-stars.

This can be interpreted as:

$$C \approx \Pr(Y_{ih}=1 \mid Y_{ij}=1,\ Y_{jh}=1)$$

thus measuring dyadic dependence.

---

## 2. Node Centrality and Network Centralisation

In the course, centrality measures (for undirected, unweighted networks) are presented using the notation $\zeta$.

### Degree Centrality

$$\zeta_i^{d} = \sum_{j\neq i} y_{ij}, \qquad i=1,\dots,n$$

It can be normalised by dividing by $(n-1)$.

### Closeness Centrality

The geodesic distance $d_{ij}$ is defined as the length of the shortest path between $i$ and $j$.
The *farness* of node $i$ is:

$$\ell_i = d_{i1}+d_{i2}+\cdots+d_{in}$$

and the closeness centrality:

$$\zeta_i^{c}=\frac{1}{\ell_i}$$

The normalised version is obtained by multiplying by $(n-1)$.

### Betweenness Centrality

$$\zeta_i^{b}=\sum_{j \neq i}\sum_{k \neq i, k > j}\frac{n_{jk}^{i}}{g_{jk}}$$

where $n_{jk}^{i}$ is the number of geodesic paths between $j$ and $k$ passing through $i$, and $g_{jk}$ is the total number of geodesic paths between $j$ and $k$. The sum runs over all pairs $(j,k)$ with $j < k$ and both different from $i$.

### Network Centralisation (Freeman, 1979)

Given $\zeta_{\max}=\max_i \zeta_i$, the network centralisation is:

$$CI=\frac{\sum_{i=1}^n (\zeta_{\max}-\zeta_i)}{\max_Y \sum_{i=1}^n (\zeta_{\max}-\zeta_i)}$$

where the denominator is the maximum sum achievable over all graphs of size $n$ (depends on the chosen index).

> **Note**: The toolkit also includes **eigenvector centrality** as a practical extension; this measure is standard in the literature but is not among those formalised in the course slides.

---

## 3. Stochastic Block Model (SBM)

The SBM is a generative model where the probability of observing a tie between two nodes depends on latent block membership variables. The probability of tie formation between network nodes is assumed to depend upon **unobserved features** of the nodes themselves. These features are captured by considering *node-specific, discrete, latent variables* that directly allow to cluster nodes (blocks).

### From Blockmodels to Stochastic Blockmodels

#### Blockmodels (BM) — Known Membership

In the original **Blockmodel** formulation:
- Each node $i = 1, \ldots, n$ belongs to one of $Q$ different blocks (classes)
- Block membership **is assumed to be known**, as well as the "true" number of blocks $Q$
- Conditional on block membership of nodes $i$ and $j$, the tie variable $Y_{ij}$ is independent of any other variable in $Y$ and follows a Bernoulli distribution
- The *success probability* (probability of observing a tie) only depends on the corresponding block membership:

$$[Y_{ij} \mid i \in q, j \in \ell] \perp\!\!\!\perp [Y_{hk} \mid h \in g, k \in m]$$

$$Y_{ij} \mid (i \in q, j \in \ell) \sim \text{Bern}(\pi_{q\ell})$$

For an undirected network, $\pi_{q\ell} = \pi_{\ell q}$.

#### Stochastic Blockmodels (SBM) — Latent Membership

The assumption that block membership is known is generally **not realistic**. The same is true about the possibility to correctly specify the "true" blocks $\{1, 2, \ldots, Q\}$. In this respect, one may consider the **stochastic counterpart** of blockmodels.

Key differences from BM:
- Blocks are **unobserved**, as well as the number of blocks $Q$
- These can be represented by means of **independent, latent variables**
- The model allows to identify clusters of nodes characterised by a *similar relational profile* (similar social behaviour)
- The SBM framework allows to define a **model-based clustering procedure**

### Latent Variables

Each node $i$ belongs to one of $Q$ blocks, represented by the indicator vector:

$$Z_i = (Z_{i1},\dots,Z_{iQ})^\top \quad\text{where}\quad Z_{iq} = \begin{cases} 1 & \text{if node } i \text{ belongs to block } q \\ 0 & \text{otherwise} \end{cases}$$

with the constraint $\sum_{q=1}^{Q} Z_{iq} = 1$ for all $i = 1, \ldots, n$ (each node belongs to exactly one block).

The $Z_i$ are i.i.d.:

$$Z_i \sim \text{Multinomial}(1,\alpha), \qquad \alpha_q=\Pr(Z_{iq}=1),\quad \sum_q \alpha_q=1$$

where $\alpha = (\alpha_1, \ldots, \alpha_Q)'$ is the vector of block proportions.

### Dyadic Model (binary network)

Conditional on block membership, the random variables $Y_{ij}$ are **independent of each other**, so that the corresponding joint probability can be factorised as:

$$\Pr(Y \mid Z) = \prod_{ij} \Pr(Y_{ij} \mid Z)$$

In particular, the univariate conditional distribution above **only** depends on the latent variables associated to the nodes $i$ and $j$:

$$\Pr(Y_{ij} \mid Z) = \Pr(Y_{ij} \mid Z_i, Z_j)$$

and, to deal with binary networks, this is assumed to correspond to a **conditional Bernoulli distribution**:

$$Y_{ij} \mid (Z_i, Z_j) \sim \text{Bern}(\pi_{q\ell}) \quad \Leftrightarrow \quad Z_{iq} = Z_{j\ell} = 1$$

That is, $\pi_{q\ell}$ denotes the probability for nodes in block $q$ to be connected with those in block $\ell$. For an undirected network, $\pi_{q\ell} = \pi_{\ell q}$.

### ML Estimation and Intractability

Let $\theta = (\alpha_1, \ldots, \alpha_{Q-1}, \pi_{11}, \ldots, \pi_{QQ})'$ denote the vector of model parameters and let $Z = \{Z_{iq}\}$ denote the matrix of all latent variables in the model.

To derive estimates $\hat{\theta}$, we may rely on a ML approach based on the following maximisation problem:

$$\hat{\theta} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \log \Pr(Y; \theta)$$

$$= \arg\max_\theta \log \sum_z \Pr(Y = y, Z = z; \theta)$$

$$= \arg\max_\theta \log \sum_z \Pr(Y = y \mid Z = z; \theta) \Pr(Z = z; \theta)$$

where:
- $z$: realisation of $Z = (Z_1, \ldots, Z_n)'$
- $\Pr(Y = y \mid Z = z) = \prod_{ij} \Pr(Y_{ij} = y_{ij} \mid Z_i = z_i, Z_j = z_j)$
- $\Pr(Z = z) = \prod_{i \leq n} \alpha' Z_i$

Solving the above problem requires the evaluation of a multiple summation defined over **all possible configurations** of the latent variables:

$$\sum_{z_1} \cdots \sum_{z_n}$$

Such a summation becomes **prohibitive** as the size of the network increases ($Q^n$ terms). For a network with 100 nodes and 5 blocks, this means $5^{100} \approx 10^{70}$ terms.

### The EM Algorithm — Why It Fails

As we are dealing with latent variables, the **EM algorithm** could be a possible choice to simplify the estimation process. In this case, instead of directly maximising $\ell(\theta)$, we start from the *complete-data log-likelihood function*:

$$\ell_c(\theta) = \log \Pr(Y = y, Z = z; \theta)$$

and alternate two steps:
- **E-step**: compute the posterior probability of latent variables, conditional on $y$ and the current estimates: $\Pr(Z = z \mid Y = y; \theta) = p(z \mid y; \theta)$
- **M-step**: maximise the expected complete-data log-likelihood with respect to model parameters $\theta$

However, performing the E-step **still requires the evaluation of the likelihood function**, which is intractable. An alternative method is needed.

### Variational EM Algorithm (VEM)

We may rely on a **variational approximation** to the log-likelihood. Instead of maximising the observed-data log-likelihood, we maximise a *computationally tractable lower bound* (auxiliary function):

$$F(q(z), \theta) = \ell(\theta) - \mathrm{KL}[q(z), p(z \mid y; \theta)]$$

where $\mathrm{KL}$ is the **Kullback-Leibler divergence** between the true intractable posterior $p(z \mid y; \theta)$ and its approximation $q(z)$.

We constrain $q(z)$ to have the following factorised form:

$$q(z) = \prod_i h(z_i; \tau_i)$$

where $h(z_i; \tau_i)$ denotes the **Multinomial distribution** with parameters 1 and $\tau_i = (\tau_{i1}, \ldots, \tau_{iQ})$.

That is, we approximate the intractable posterior distribution $p(z \mid y; \theta)$ by assuming **independence a posteriori** between latent variables $Z_i$'s. Therefore, the quantity $\tau_{iq}$ can be directly considered as an approximation of $\Pr(Z_{iq} = 1 \mid Y = y)$.

The VE algorithm alternates two separate steps:

1. **VE-step**: we maximise $F(q(z), \theta)$ with respect to $q(z)$. That is, we maximise $F(q(z), \theta)$ with respect to the $\tau_i$ parameters (keeping constant the remaining model parameters)

2. **ME-step**: we maximise $F(q(z), \theta)$ with respect to $\alpha_q$, $q = 1, \ldots, Q$ and $\pi_{q\ell}$, $q, \ell = 1, \ldots, Q$ (keeping constant $\tau_i$, $i = 1, \ldots, n$ to the current estimates)

Starting from some initial values for $\tau_i^{(0)}$, the VE-step and ME-step are alternated until **convergence** of the algorithm. That is, until the difference between two subsequent values of the auxiliary function $F(q, \theta)$ is lower than a fixed small quantity $\epsilon$:

$$F(\hat{q}^{(k)}, \theta^{(k)}) - F(\hat{q}^{(k-1)}, \theta^{(k-1)}) < \epsilon$$

### Block Assignment

At convergence, node $i$ is assigned to the most probable block:

$$\hat{q}_i = \arg\max_q \hat\tau_{iq}$$

**Clustering of nodes** is a crucial aspect in this framework. How can we assign nodes to one of the $Q$ distinct blocks? In model-based clustering, typically, statistical units are assigned to the mixture component characterised by the **highest posterior probability**. As posteriors are not available when dealing with SBM (they are intractable), we rely on their approximation $\tau_{iq}$.

That is, we assign node $i$ to one of the $Q$ blocks by solving:

$$\arg\max_q \hat{\tau}_{iq}, \quad i = 1, \ldots, n$$

where $\hat{\tau}_{iq}$ is the approximation to $\Pr(Z_{iq} = 1 \mid Y = y)$ obtained at convergence of the VEM algorithm.

### Model Selection (ICL)

The **number of blocks $Q$ is unknown** and should be estimated along with the remaining model parameters. Typically, $Q$ is treated as known and estimated via **model selection techniques** through penalised likelihood criteria.

In this framework, the likelihood is not available in closed form expression, so that an approximate approach is considered. This is based on the **Integrated Classification Likelihood (ICL)**:

$$ICL = \log p(y, \tilde{z})$$

where $\tilde{z}$ denotes the predicted $Z$ (the MAP assignment of nodes to blocks). The optimal number of latent blocks $Q$ is the one corresponding to the **maximum value of ICL**.

In practice, a **BIC-like penalised version** is often used to avoid overfitting:

$$ICL_{\text{pen}} = \log p(y,\tilde z) - \frac{Q-1}{2}\log n - \frac{Q(Q+1)}{4}\log\frac{n(n-1)}{2}$$

where the penalty terms correct for the number of free parameters in $\alpha$ (block proportions) and $\pi$ (connection probabilities).

> **Implementation note**: The toolkit computes **both** MDL (via `graph-tool`'s MCMC inference, used for model fitting) **and** the penalised ICL (as defined above, for comparison with course material). Both are reported in the output.
>
> *Note on inference method*: The course covers Variational EM for SBM inference. This toolkit uses `graph-tool`'s MCMC implementation, which achieves similar results via Markov Chain Monte Carlo sampling rather than variational approximation.

### Extensions to Valued Networks

The SBM described so far may be extended to deal with **valued relations**. Random variables $Y_{ij}$ are still assumed to be conditionally independent, given the latent variables of nodes involved in the relation. However, rather than considering a conditional Bernoulli distribution, we may have:

| Network Type | Distribution | Parameters |
|--------------|--------------|------------|
| **Binary** | $Y_{ij} \mid Z_{iq}=1, Z_{j\ell}=1 \sim \text{Bern}(\pi_{q\ell})$ | $\pi_{q\ell}$ = connection probability |
| **Real-valued** | $Y_{ij} \mid Z_{iq}=1, Z_{j\ell}=1 \sim \mathcal{N}(\mu_{q\ell}, \sigma_{q\ell})$ | $\mu_{q\ell}$ = mean, $\sigma_{q\ell}$ = std |
| **Count data** | $Y_{ij} \mid Z_{iq}=1, Z_{j\ell}=1 \sim \text{Pois}(\lambda_{q\ell})$ | $\lambda_{q\ell}$ = rate |

Further distributions may be adopted depending on the nature of the edge weights.

---

## 4. SBM vs Modularity-based Methods

A common alternative to SBM is **modularity optimisation** (e.g., Louvain algorithm). Modularity is defined as:

$$Q = \frac{1}{2m}\sum_{ij}\left(y_{ij} - \frac{k_i k_j}{2m}\right)\delta(c_i, c_j)$$

where:
- $m$ is the total number of edges
- $k_i$ is the degree of node $i$
- $c_i$ is the community assignment of node $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$, 0 otherwise

| Aspect | SBM | Modularity (e.g., Louvain) |
|--------|-----|----------------------------|
| **Approach** | Generative model | Heuristic optimisation |
| **Inference** | Probabilistic (Bayesian/VEM) | Greedy algorithm |
| **Structure detected** | Assortative AND disassortative | Only assortative |
| **Model selection** | Principled (ICL/MDL) | Resolution limit problem |
| **Uncertainty** | Provides posterior probabilities | Point estimate only |

The SBM can detect:
- **Assortative structure**: nodes connect within their block (communities)
- **Disassortative structure**: nodes connect between blocks (bipartite-like)
- **Core-periphery**: central blocks connect to all, peripheral blocks connect only to core

---

## Assortativity Coefficient

The toolkit distinguishes between two types of assortativity:

### 1. Attribute Assortativity (True Homophily)

When **external node attributes** are available (e.g., school class labels in the 5-column format `t n1 n2 c1 c2`), the toolkit computes the **true assortativity coefficient** based on the mixing matrix:

$$r = \frac{\text{Tr}(e) - \sum_i a_i^2}{1 - \sum_i a_i^2} = \frac{Q}{Q_{max}}$$

where:
- $e_{ij}$ = fraction of edges connecting nodes of class $i$ to class $j$
- $a_i = \sum_j e_{ij}$ = fraction of edge endpoints in class $i$
- $\text{Tr}(e) = \sum_i e_{ii}$ = fraction of edges within the same class

This measures **real homophily**: do nodes preferentially connect to others with the same external attribute?

**Interpretation**:
- $r = 1$: Perfect assortative mixing (all edges within classes)
- $r = 0$: Random mixing (no preference for same-class connections)
- $r < 0$: Disassortative mixing (preference for different-class connections)

### 2. Partition Modularity (SBM Quality)

When computing modularity based on **SBM-inferred blocks** (not external attributes), we get a measure of partition quality:

$$Q_{partition} = \frac{1}{2m}\sum_{ij}\left(Y_{ij} - \frac{k_i k_j}{2m}\right)\delta(b_i, b_j)$$

where $b_i$ is the block assignment from SBM inference.

> **Important**: These are different concepts! Attribute assortativity measures true homophily based on known labels; partition modularity measures the quality of the inferred partition.
>
> For the LyonSchool dataset:
> - **Attribute assortativity** (11 classes): $r = 0.2338$ (moderate homophily)
> - **Partition modularity** (18 SBM blocks): $Q/Q_{max} = 0.134$ (lower due to finer granularity — 18 blocks vs 11 classes)

---

## 5. Temporal Extension (Dynamic SBM) — Project Extension

For temporal networks constructed from edge lists with timestamps, we adopt a sliding-window strategy:

1. Segment time into overlapping windows
2. For each window $w$, construct a snapshot $y^{(w)}$
3. Fit an independent SBM for each $w$, obtaining $\hat\tau^{(w)}$ / partitions
4. Solve the *label switching* problem by aligning labels across consecutive windows (Hungarian algorithm on block overlaps)
5. Compute transition and stability statistics

**This part is a project extension** (not formalised in the course slides), but maintains the same logic of latent partitioning applied to snapshots.

### Transition Matrix

After alignment, we compute the transition probability matrix:

$$P_{rs} = P(b_i^{(t+1)} = s \mid b_i^{(t)} = r) = \frac{|\{i : b_i^{(t)} = r \land b_i^{(t+1)} = s\}|}{|\{i : b_i^{(t)} = r\}|}$$

**Interpretation**:
- **High diagonal values**: Stable communities (nodes stay in their blocks)
- **Off-diagonal values**: Community fluidity (nodes moving between blocks)

### Block Stability

For each block $r$:

$$\text{Stability}(r) = P_{rr} = P(b^{(t+1)} = r \mid b^{(t)} = r)$$

### Mobile Nodes

Nodes that frequently change blocks:

$$\text{Mobility}(i) = \frac{|\{t : b_i^{(t)} \neq b_i^{(t+1)}\}|}{T - 1}$$

High-mobility nodes may represent:
- Bridge individuals connecting different communities
- Individuals with diverse social circles
- Temporal visitors (e.g., teachers moving between classes)

---

## 6. Hypergraph Group Extraction via Cliques

Many real-world social interactions involve more than two individuals simultaneously (e.g., group conversations, classroom activities, meetings). Standard pairwise network representations may fail to capture these **higher-order interactions**.

### From Pairwise to Group Interactions

Given a temporal network recorded as pairwise contacts $(t, i, j)$, we can approximate group interactions using **clique extraction** (Iacopini et al., 2022):

1. For each time window $[t, t + \Delta t)$, aggregate all contacts into a snapshot graph $G_w$
2. Extract **maximal cliques** from $G_w$
3. Interpret each clique as a group interaction (hyperedge)

A **clique** is a complete subgraph where every pair of nodes is connected. If three individuals all interacted pairwise within the same time window, they form a triangle (clique of size 3), suggesting a group interaction.

### Maximal Cliques

A **maximal clique** is a clique that cannot be extended by adding any adjacent vertex. This ensures we capture the largest groups without redundancy.

### Group Size Distribution

The distribution of group sizes $P(k)$ reveals the social structure:
- **Power-law decay**: Many small groups, few large ones (typical in face-to-face data)
- **Characteristic scale**: Preferred group sizes (e.g., classrooms, teams)

$$P(k) = \frac{|\{C : |C| = k\}|}{\sum_k |\{C : |C| = k\}|}$$

where $C$ denotes a clique (group) and $|C|$ its size.

### Computational Considerations

Maximal clique enumeration can be computationally expensive (exponential in worst case). The toolkit implements safety measures:
- Maximum clique size limit (`max_group_size`)
- Maximum cliques per window (`max_cliques_per_window`)
- Density-based skipping for overly dense graphs

---

## References

For the complete reference list, see the [References section in README.md](README.md#references).

### Key Course References

- Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels: First steps. *Social Networks*, *5*(2), 109–137.
- Freeman, L. C. (1979). Centrality in social networks: Conceptual clarification. *Social Networks*, *1*(3), 215–239.
- Nowicki, K., & Snijders, T. A. B. (2001). Estimation and prediction for stochastic blockstructures. *Journal of the American Statistical Association*, *96*(455), 1077–1087.
- Daudin, J. J., Picard, F., & Robin, S. (2008). A mixture model for random graphs. *Statistics and Computing*, *18*(2), 173–183.

### Course Material

Marino, M. F. (2024–2025). *Network Data Analysis* [Lecture slides]. Master's Degree in Data Science and Statistical Learning (MD2SL), Dipartimento di Statistica, Informatica, Applicazioni (DiSIA), Università degli Studi di Firenze.

---

*Last updated: January 2026*
