# Temporal Network SBM

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![graph-tool](https://img.shields.io/badge/graph--tool-2.59-green.svg)](https://graph-tool.skewed.de/)

A comprehensive Python toolkit for analyzing **temporal (dynamic) networks** using **Stochastic Block Models (SBM)**.

This project was developed as an examination work for the course Network Data Analysis taught by Prof. Maria Francesca Marino at the University of Florence, within the Master's program in Data Science and Statistical Learning MD2SL.

The toolkit implements rigorous statistical methods for network community detection, combining classical network analysis with modern inference-based approaches. It provides a complete pipeline from raw temporal edge data to publication-quality visualizations and comprehensive statistical reports.

![Network Animation Example](docs/network_animation.gif)

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Input Data Format](#input-data-format)
5. [Output](#output)
6. [Example: LyonSchool Dataset](#example-lyonschool-dataset)
7. [Configuration](#configuration)
8. [Theoretical Background](#theoretical-background)
9. [References](#references)
10. [Acknowledgments](#acknowledgments)
11. [License](#license)

---

## Features

This toolkit provides a comprehensive suite of network analysis methods, structured around three main pillars: static network analysis, stochastic block model inference, and temporal dynamics.

### Static Network Analysis

The foundation of any network analysis begins with understanding the basic structural properties:

- **Global statistics**: nodes, edges, density, connected components
- **Distance metrics**: diameter, average path length
- **Degree analysis**: distribution, moments (mean, std, min, max, median)
- **Clustering**: local clustering coefficient, global transitivity
- **Centrality measures**: 
  - Degree centrality
  - Betweenness centrality
  - Closeness centrality  
  - Eigenvector centrality
- **Network centralization** using Freeman's index

### Stochastic Block Model (SBM)

Moving beyond heuristic community detection, this toolkit uses principled Bayesian inference:

- Automatic detection of optimal number of blocks via **Minimum Description Length (MDL)**
- Block membership assignment with uncertainty quantification
- Inter-block connection probability matrix Θ
- Internal block density analysis
- Comparison with degree-corrected variants

### Dynamic Stochastic Block Model

For temporal networks, we implement a sliding-window approach inspired by the `dynsbm` methodology:

- Partition temporal data into overlapping time windows
- Independent SBM fits per window
- **Label alignment** across windows using the Hungarian algorithm
- **Transition probability matrix** P(b_{t+1} | b_t)
- Block stability metrics
- Identification of **mobile nodes** (nodes frequently changing blocks)

### Visualization Suite

All analyses are accompanied by publication-ready visualizations:

- Degree distribution (linear + log-log scale)
- Centrality comparison scatter plots
- Network graph with centrality-based node sizing
- SBM community structure visualization
- Block connection probability heatmap
- Temporal activity timeline
- Dynamic SBM transition heatmap
- Block evolution over time
- **Animated network evolution** (optional, 4K resolution)

---

## Installation

### System Requirements

- **Python 3.8+**
- **Linux or WSL** (Windows Subsystem for Linux) — required for graph-tool
- 4GB RAM recommended for large networks

### Step 1: Install graph-tool

The core of this toolkit relies on `graph-tool`, a highly efficient C++ library for network analysis. It cannot be installed via pip.

**Option A: Conda (recommended for most users)**
```bash
conda create -n netsbm python=3.10
conda activate netsbm
conda install -c conda-forge graph-tool
```

**Option B: Ubuntu/Debian via apt (recommended for WSL)**
```bash
sudo apt update
sudo apt install python3-graph-tool
```

**Option C: Docker**
```bash
docker pull tiagopeixoto/graph-tool
docker run -it tiagopeixoto/graph-tool python3
```

### Step 2: Install Python dependencies

```bash
pip install numpy scipy matplotlib pyyaml
```

Or using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Clone this repository

```bash
git clone https://github.com/yourusername/temporal-network-sbm.git
cd temporal-network-sbm
```

### Windows Users

This toolkit requires **WSL** (Windows Subsystem for Linux):

1. Open PowerShell as Administrator and run: `wsl --install`
2. Install Ubuntu 24.04 from Microsoft Store
3. Inside WSL, install graph-tool: `sudo apt install python3-graph-tool`
4. Run the toolkit from within WSL

---

## Quick Start

### Basic Usage

```bash
python main.py --input &lt;data_file&gt; --output &lt;output_dir&gt;
```

### Example Commands

```bash
# Full analysis with default settings
python main.py --input data/contacts.dat --output results/

# With custom configuration
python main.py --input data.dat --output results/ --config my_config.yaml

# Generate network animation (resource-intensive)
python main.py --input data.dat --output results/ --animate

# Skip dynamic SBM for faster execution
python main.py --input data.dat --output results/ --no-dynamic-sbm
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--input` | `-i` | Path to input data file (required) |
| `--output` | `-o` | Path to output directory (required) |
| `--config` | `-c` | Path to YAML configuration file |
| `--animate` | | Generate network animation (MP4/GIF) |
| `--no-dynamic-sbm` | | Skip dynamic SBM analysis |

---

## Input Data Format

The toolkit is designed to be **generic** — it works with any temporal edge list, not hardcoded to specific datasets.

### Expected Format

A text file with three columns:

```
timestamp    node1    node2
31220        1        42
31220        5        12
31240        1        42
...
```

### Column Specification

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Integer | Timestamp (seconds, UNIX time, or any integer) |
| `node1` | Integer | ID of the first node |
| `node2` | Integer | ID of the second node |

### Separator Support

The parser auto-detects common separators:
- Space (default)
- Tab (`\t`)
- Comma (`,`)
- Semicolon (`;`)

You can also specify the separator explicitly in `config.yaml`.

### Compatible Datasets

This format is compatible with many public network datasets:

- **SocioPatterns** (LyonSchool, SFHH, InVS, Thiers, Hospital, etc.)
- **Socio-economic networks** (email-EU, Congress bills)
- Any temporal edge list with the above structure

---

## Output

The toolkit generates a comprehensive analysis package:

```
output/
├── figures/
│   ├── degree_distribution.png
│   ├── centrality_comparison.png
│   ├── centrality_network.png
│   ├── community_sbm.png
│   ├── sbm_block_matrix.png
│   ├── temporal_activity.png
│   ├── dynamic_sbm_transitions.png
│   ├── dynamic_sbm_evolution.png
│   └── network_animation.mp4  (if --animate)
├── metrics.csv
├── top_nodes.csv
├── sbm_results.csv
├── dynamic_sbm_windows.csv
├── dynamic_sbm_transitions.csv
├── dynamic_sbm_stability.csv
└── summary.txt
```

### Output File Descriptions

| File | Description |
|------|-------------|
| `metrics.csv` | All network-level metrics in machine-readable format |
| `top_nodes.csv` | Top 20 nodes ranked by each centrality measure |
| `sbm_results.csv` | Block assignments, sizes, and internal densities |
| `dynamic_sbm_windows.csv` | Per-window SBM results (blocks, edges, MDL) |
| `dynamic_sbm_transitions.csv` | Block-to-block transition probability matrix |
| `dynamic_sbm_stability.csv` | Stability score for each block |
| `summary.txt` | Human-readable comprehensive report |

---

## Example: LyonSchool Dataset

To demonstrate the toolkit's capabilities, we analyze the **LyonSchool** dataset from the [SocioPatterns project](http://www.sociopatterns.org/).

### About the Dataset

The LyonSchool dataset captures face-to-face interactions between students and teachers in a primary school in Lyon, France, recorded using RFID sensors over two consecutive school days.

| Property | Value |
|----------|-------|
| Nodes | 242 (students + teachers) |
| Temporal edges | 125,773 |
| Unique static edges | 8,317 |
| Duration | ~2 days (~32 hours) |
| Time resolution | 20 seconds |

### Running the Analysis

```bash
python main.py \
  --input /path/to/tij_LyonSchool.dat \
  --output results_lyonschool/
```

### Results Summary

Below are the complete results obtained from our analysis pipeline.

---

#### Network Statistics

| Metric | Value |
|--------|-------|
| **Nodes** | 242 |
| **Edges** | 8,317 |
| **Density** | 0.2852 |
| **Connected** | Yes (single component) |
| **Diameter** | 3 |
| **Average Path Length** | 1.732 |

The network is **dense** (28.5% of possible edges exist) and **highly connected** — any two individuals can reach each other in at most 3 hops, with an average of less than 2.

---

#### Degree Distribution

| Statistic | Value |
|-----------|-------|
| Mean degree | 68.74 |
| Std deviation | 26.57 |
| Min degree | 20 |
| Max degree | 134 |
| Median | 68.5 |

![Degree Distribution](docs/degree_distribution.png)

The degree distribution is **unimodal and approximately symmetric**, suggesting relatively homogeneous participation across individuals. The log-log plot shows the network does not follow a power-law distribution, which is typical for social contact networks in closed environments (schools, workplaces).

---

#### Clustering Analysis

| Metric | Value |
|--------|-------|
| **Clustering Coefficient** | 0.5255 |
| **Transitivity** | 0.4798 |

The high clustering coefficient (52.5%) indicates strong local cohesion — if person A interacts with B and C, there's a high probability that B and C also interact. This is characteristic of school environments where students form tight-knit groups.

---

#### Centrality Analysis

![Centrality Comparison](docs/centrality_comparison.png)

The scatter plot reveals **strong correlation between degree and eigenvector centrality**, meaning well-connected individuals are connected to other well-connected individuals — a signature of core-periphery structure.

![Network with Centrality](docs/centrality_network.png)

Network visualization with node sizes proportional to degree centrality. The graph reveals a relatively dense core with some peripheral nodes.

---

#### Network Centralization (Freeman's Index)

| Centralization Measure | Value | Interpretation |
|------------------------|-------|----------------|
| **Degree** | 0.2731 | Moderately centralized |
| **Betweenness** | 0.0000 | Decentralized |
| **Closeness** | 0.1128 | Slightly centralized |

The network is **slightly centralized** overall. The near-zero betweenness centralization is notable — it means no single node dominates shortest paths, which is expected in a dense network with short diameter. The moderate degree centralization indicates some individuals are more socially active than others.

---

#### Top 10 Central Nodes

| Rank | Node ID | Degree Centrality |
|------|---------|-------------------|
| 1 | 1551 | 0.556 |
| 2 | 1780 | 0.535 |
| 3 | 1761 | 0.531 |
| 4 | 1673 | 0.514 |
| 5 | 1665 | 0.510 |
| 6 | 1552 | 0.510 |
| 7 | 1579 | 0.506 |
| 8 | 1700 | 0.502 |
| 9 | 1890 | 0.498 |
| 10 | 1765 | 0.490 |

These individuals had contact with more than 50% of the school population during the observation period.

---

#### Stochastic Block Model Results

![SBM Communities](docs/community_sbm.png)

The SBM infers the latent community structure using Bayesian inference with the Minimum Description Length criterion.

| SBM Parameter | Value |
|---------------|-------|
| **Optimal number of blocks** | 18 |
| **Description Length (MDL)** | 15,827.51 bits |
| **ICL** | -11,510.71 |

> **Note**: The toolkit computes both MDL (from `graph-tool`) and ICL (following course notation). MDL is used for model selection during inference; ICL is reported for comparison with the course material.

##### Block Sizes and Densities

| Block | Size | Internal Density |
|-------|------|------------------|
| Block 1 | 8 | 0.857 |
| Block 2 | 11 | 1.000 |
| Block 3 | 11 | 0.946 |
| Block 4 | 15 | 1.000 |
| Block 5 | 13 | 1.000 |
| Block 6 | 16 | 0.983 |
| Block 7 | 23 | 0.968 |
| Block 8 | 16 | 0.950 |
| Block 9 | 11 | 1.000 |
| Block 10 | 16 | 0.992 |
| Block 11 | 16 | 0.992 |
| Block 12 | 20 | 0.805 |
| Block 13 | 8 | 1.000 |
| Block 14 | 11 | 0.982 |
| Block 15 | 8 | 1.000 |
| Block 16 | 17 | 0.993 |
| Block 17 | 6 | 1.000 |
| Block 18 | 16 | 0.983 |

The 18 blocks likely correspond to **school classes** — the internal density close to 1.0 indicates that students within the same class interact with almost everyone else in their class.

![Block Connection Matrix](docs/sbm_block_matrix.png)

*Figure: Connection Probability Matrix $\hat{\Pi}$, where $\hat{\pi}_{ql}$ represents the estimated probability of an edge between a node in block $q$ and a node in block $\ell$.*

The block connection matrix shows:
- **Strong diagonal** (assortative structure): students primarily interact within their class
- **Off-diagonal connections**: inter-class interactions during breaks, lunch, etc.

---

#### Temporal Analysis

![Temporal Activity](docs/temporal_activity.png)

| Temporal Metric | Value |
|-----------------|-------|
| **Duration** | 1,948.3 minutes (~32.5 hours) |
| **Unique timestamps** | 3,100 |
| **Time windows** | 1,039 |
| **Average edges/window** | 265.1 |
| **Peak activity** | 554 edges |

The temporal activity shows clear **periodic patterns**:
- Peaks during school hours (class activities, breaks)
- Valleys during night hours
- Two main activity clusters corresponding to the two school days

---

#### Dynamic SBM Results

The Dynamic SBM analyzes how community structure evolves over time by fitting independent SBMs to each time window and aligning labels across consecutive windows.

![Dynamic SBM Evolution](docs/dynamic_sbm_evolution.png)

*Figure: Block Size Heatmap showing the number of nodes in each block (rows) across time windows (columns). Darker cells indicate larger blocks; white cells indicate absent blocks.*

**Interpretation notes:**
- Each row represents a block label (0, 1, 2, ...) as assigned after Hungarian algorithm alignment
- The number of active blocks varies between 1 and 7 across time windows
- **Low activity periods** (windows ~17–27, corresponding to night): 1–3 blocks detected
- **High activity periods** (school hours): 5–7 blocks detected

> **Caveat on label alignment**: The Hungarian algorithm aligns labels locally between consecutive windows $(t, t+1)$, which can cause "label drift" over longer periods. A block that appears as "Block 0" at $t=0$ may be relabeled as "Block 3" by $t=20$ due to accumulated misalignments. The transition matrix and heatmap should be interpreted with this limitation in mind.

![Dynamic SBM Transitions](docs/dynamic_sbm_transitions.png)

*Figure: Transition Probability Matrix $\hat{P}$, where $\hat{p}_{rs} = P(b_{t+1}=s \mid b_t=r)$ represents the estimated probability that a node moves from block $r$ at time $t$ to block $s$ at time $t+1$.*

The transition matrix shows block-to-block movement probabilities. Key observations:
- **Diagonal dominance**: nodes tend to stay in their blocks (stable class membership)
- **Some off-diagonal flow**: students occasionally interact with other classes
- **Block stability ranges from 35% to 80%** depending on the class

| Windows Analyzed | 50 |
|------------------|-----|
| Block stability range | 0.35 – 0.80 |
| Nodes that changed blocks | 242 (all) |

All 242 nodes changed blocks at least once, reflecting the natural dynamics of school life — students temporarily join different groups during breaks, lunch, or cross-class activities.

---

## Configuration

Customize the analysis by editing `config.yaml`:

```yaml
# Input file parsing
input:
  separator: "auto"      # auto, space, tab, comma
  skip_header: false

# Time window parameters  
time_windows:
  window_size_seconds: 300   # 5 minutes
  window_step_seconds: 60    # 1 minute overlap

# Static SBM parameters
sbm:
  max_blocks: 20
  equilibrate: true          # More accurate but slower

# Dynamic SBM parameters
dynamic_sbm:
  enabled: true
  max_windows: 50            # Number of windows to analyze
  max_blocks: 10             # Max blocks per window

# Visualization settings
visualization:
  dpi: 300
  format: "png"

# Animation settings
animation:
  enabled: false
  max_frames: 100
  fps: 10
  resolution: [1920, 1080]   # HD, use [3840, 2160] for 4K
```

---

## Theoretical Background

This section summarizes the mathematical foundations used in the toolkit, following the notation from the *Network Data Analysis* course (M.F. Marino).

### Basic Notation

We consider a network on $n$ nodes $\{1,\dots,n\}$ described by the random adjacency matrix $Y$ and its observed realization $y$.
The element $Y_{ij}$ is the dyadic variable representing the relationship between nodes $i$ and $j$.  
In the binary case, $Y_{ij} = 1$ if there is an edge between $i$ and $j$, and $Y_{ij} = 0$ otherwise.

For undirected networks, symmetry holds: $Y_{ij}=Y_{ji}$; for directed networks, $Y_{ij}$ and $Y_{ji}$ may differ.

---

### 1. Global Statistics: Density, Reciprocity, Transitivity

#### Density

The density $\rho$ is the ratio between observed ties and possible ties:

$$\rho = \frac{m}{m_{\max}}$$

where $m$ is the number of observed ties and $m_{\max}$ is the maximum possible number of ties.

For directed networks, the denominator is $n(n-1)$; for undirected networks, $n(n-1)/2$.
In practice, starting from $y$, the numerator is computed as $\sum_i\sum_{j\neq i} y_{ij}$ (directed) or $\sum_i\sum_{j<i} y_{ij}$ (undirected).
Moreover, $\rho \approx \Pr(Y_{ij}=1)$, i.e., it estimates the probability of observing a tie between two randomly selected nodes.

#### Reciprocity (directed networks)

Reciprocity $R$ is the fraction of reciprocated ties:

$$R = \frac{\sum_{ij} y_{ij}y_{ji}}{m}$$

where $m$ is the number of observed ties.

#### Transitivity / Clustering

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

### 2. Node Centrality and Network Centralization

In the course, centrality measures (for undirected, unweighted networks) are presented using the notation $\zeta$.

#### Degree Centrality

$$\zeta_i^{d} = \sum_{j\neq i} y_{ij}, \qquad i=1,\dots,n$$

It can be normalized by dividing by $(n-1)$.

#### Closeness Centrality

The geodesic distance $d_{ij}$ is defined as the length of the shortest path between $i$ and $j$.
The *farness* of node $i$ is:

$$\ell_i = d_{i1}+d_{i2}+\cdots+d_{in}$$

and the closeness centrality:

$$\zeta_i^{c}=\frac{1}{\ell_i}$$

The normalized version is obtained by multiplying by $(n-1)$.

#### Betweenness Centrality

$$\zeta_i^{b}=\sum_{j>i}\sum_{k>j}\frac{n_{jk}^{i}}{g_{jk}}$$

where $n_{jk}^{i}$ is the number of geodesic paths between $j$ and $k$ passing through $i$, and $g_{jk}$ is the total number of geodesic paths between $j$ and $k$.

#### Network Centralization (Freeman, 1979)

Given $\zeta_{\max}=\max_i \zeta_i$, the network centralization is:

$$CI=\frac{\sum_{i=1}^n (\zeta_{\max}-\zeta_i)}{\max_Y \sum_{i=1}^n (\zeta_{\max}-\zeta_i)}$$

where the denominator is the maximum sum achievable over all graphs of size $n$ (depends on the chosen index).

> **Note**: The toolkit also includes **eigenvector centrality** as a practical extension; this measure is standard in the literature but is not among those formalized in the course slides.

---

### 3. Stochastic Block Model (SBM)

The SBM is a generative model where the probability of observing a tie between two nodes depends on latent block membership variables.

#### Latent Variables

Each node $i$ belongs to one of $Q$ blocks, represented by the indicator vector:

$$Z_i = (Z_{i1},\dots,Z_{iQ})^\top \quad\text{where}\quad Z_{iq} = 1 \text{ if node } i \in \text{block } q, \quad 0 \text{ otherwise}, \quad \sum_q Z_{iq}=1$$

The $Z_i$ are i.i.d.:

$$Z_i \sim \text{Multinomial}(1,\alpha), \qquad \alpha_q=\Pr(Z_{iq}=1),\quad \sum_q \alpha_q=1$$

#### Dyadic Model (binary network)

Conditionally on block memberships, dyads are independent:

$$\Pr(Y\mid Z)=\prod_{ij}\Pr(Y_{ij}\mid Z), \qquad \Pr(Y_{ij}\mid Z)=\Pr(Y_{ij}\mid Z_i,Z_j)$$

with:

$$Y_{ij}\mid(Z_{iq}=1,Z_{j\ell}=1)\sim \text{Bern}(\pi_{q\ell})$$

For undirected networks, $\pi_{q\ell}=\pi_{\ell q}$.

#### Estimation and Intractability

Denoting $\theta=(\alpha,\pi)$ as the parameters, ML estimation requires:

$$\hat\theta=\arg\max_\theta \log \Pr(Y;\theta) =\arg\max_\theta \log \sum_z \Pr(Y=y,Z=z;\theta)$$

but the sum over all latent configurations $z$ grows prohibitively ($Q^n$ terms).

#### Variational EM (VEM)

A variational approximation of the log-likelihood is used:

$$F(q(z),\theta)=\ell(\theta)-\mathrm{KL}\big[q(z),p(z\mid y;\theta)\big]$$

A factorization is imposed:

$$q(z)=\prod_i h(z_i;\tau_i), \qquad \tau_i=(\tau_{i1},\dots,\tau_{iQ})$$

where $\tau_{iq}$ approximates $\Pr(Z_{iq}=1\mid Y=y)$.

The algorithm alternates:
1. **VE-step**: optimize $F$ with respect to $\tau_i$  
2. **ME-step**: optimize $F$ with respect to $\alpha$ and $\pi$, with $\tau$ fixed

until convergence.

#### Block Assignment

At convergence, node $i$ is assigned to the most probable block:

$$\hat{q}_i = \arg\max_q \hat\tau_{iq}$$

#### Model Selection (ICL)

The number of blocks $Q$ can be selected via the **Integrated Classification Likelihood**. In the course slides, the ICL is presented as:

$$ICL = \log p(y, \tilde{z})$$

where $\tilde{z}$ is the MAP (maximum a posteriori) assignment of nodes to blocks. The number $Q$ is chosen to maximize this criterion.

In practice, a **BIC-like penalized version** is often used to avoid overfitting:

$$ICL_{\text{pen}} = \log p(y,\tilde z) - \frac{Q-1}{2}\log n - \frac{Q(Q+1)}{4}\log\frac{n(n-1)}{2}$$

where the penalty terms correct for the number of free parameters in $\alpha$ (block proportions) and $\pi$ (connection probabilities).

> **Implementation note**: The toolkit computes **both** MDL (via `graph-tool`'s MCMC inference, used for model fitting) **and** the penalized ICL (as defined above, for comparison with course material). Both are reported in the output.

#### Extensions to Valued Networks

For non-binary relationships:
- (real-valued) $Y_{ij}\mid(Z_{iq}=1,Z_{j\ell}=1)\sim \mathcal{N}(\mu_{q\ell},\sigma_{q\ell})$
- (count) $Y_{ij}\mid(Z_{iq}=1,Z_{j\ell}=1)\sim \mathrm{Pois}(\lambda_{q\ell})$

---

### 4. SBM vs Modularity-based Methods

A common alternative to SBM is **modularity optimization** (e.g., Louvain algorithm). Modularity is defined as:

$$Q = \frac{1}{2m}\sum_{ij}\left(y_{ij} - \frac{k_i k_j}{2m}\right)\delta(c_i, c_j)$$

where:
- $m$ is the total number of edges
- $k_i$ is the degree of node $i$
- $c_i$ is the community assignment of node $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$, 0 otherwise

| Aspect | SBM | Modularity (e.g., Louvain) |
|--------|-----|----------------------------|
| **Approach** | Generative model | Heuristic optimization |
| **Inference** | Probabilistic (Bayesian/VEM) | Greedy algorithm |
| **Structure detected** | Assortative AND disassortative | Only assortative |
| **Model selection** | Principled (ICL/MDL) | Resolution limit problem |
| **Uncertainty** | Provides posterior probabilities | Point estimate only |

The SBM can detect:
- **Assortative structure**: nodes connect within their block (communities)
- **Disassortative structure**: nodes connect between blocks (bipartite-like)
- **Core-periphery**: central blocks connect to all, peripheral blocks connect only to core

---

### 5. Temporal Extension (Dynamic SBM) — Project Extension

For temporal networks constructed from edge lists with timestamps, we adopt a sliding-window strategy:

1. Segment time into overlapping windows
2. For each window $w$, construct a snapshot $y^{(w)}$
3. Fit an independent SBM for each $w$, obtaining $\hat\tau^{(w)}$ / partitions
4. Solve the *label switching* problem by aligning labels across consecutive windows (Hungarian algorithm on block overlaps)
5. Compute transition and stability statistics

**This part is a project extension** (not formalized in the course slides), but maintains the same logic of latent partitioning applied to snapshots.

#### Transition Matrix

After alignment, we compute the transition probability matrix:

$$P_{rs} = P(b_i^{(t+1)} = s \mid b_i^{(t)} = r) = \frac{|\{i : b_i^{(t)} = r \land b_i^{(t+1)} = s\}|}{|\{i : b_i^{(t)} = r\}|}$$

**Interpretation**:
- **High diagonal values**: Stable communities (nodes stay in their blocks)
- **Off-diagonal values**: Community fluidity (nodes moving between blocks)

#### Block Stability

For each block $r$:

$$\text{Stability}(r) = P_{rr} = P(b^{(t+1)} = r \mid b^{(t)} = r)$$

#### Mobile Nodes

Nodes that frequently change blocks:

$$\text{Mobility}(i) = \frac{|\{t : b_i^{(t)} \neq b_i^{(t+1)}\}|}{T - 1}$$

High-mobility nodes may represent:
- Bridge individuals connecting different communities
- Individuals with diverse social circles
- Temporal visitors (e.g., teachers moving between classes)

---

## References

### Foundational Works

Holland, P. W., & Leinhardt, S. (1976). Local structure in social networks. *Sociological Methodology*, *7*, 1–45.

Holland, P. W., & Leinhardt, S. (1981). An exponential family of probability distributions for directed graphs. *Journal of the American Statistical Association*, *76*(373), 33–50. https://doi.org/10.1080/01621459.1981.10477598

Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels: First steps. *Social Networks*, *5*(2), 109–137.

### Centrality

Freeman, L. C. (1979). Centrality in social networks: Conceptual clarification. *Social Networks*, *1*(3), 215–239. https://doi.org/10.1016/0378-8733(78)90021-7

### Latent Variable Models

Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, *97*(460), 1090–1098. https://doi.org/10.1198/016214502388618906

### Stochastic Block Models

Nowicki, K., & Snijders, T. A. B. (2001). Estimation and prediction for stochastic blockstructures. *Journal of the American Statistical Association*, *96*(455), 1077–1087. https://doi.org/10.1198/016214501753208735

Daudin, J. J., Picard, F., & Robin, S. (2008). A mixture model for random graphs. *Statistics and Computing*, *18*(2), 173–183. https://doi.org/10.1007/s11222-007-9060-7

Airoldi, E. M., Blei, D. M., Fienberg, S. E., & Xing, E. P. (2008). Mixed membership stochastic blockmodels. *Journal of Machine Learning Research*, *9*, 1981–2014. https://jmlr.org/papers/v9/airoldi08a.html

### Exponential Random Graph Models

Frank, O., & Strauss, D. (1986). Markov graphs. *Journal of the American Statistical Association*, *81*(395), 832–842. https://doi.org/10.1080/01621459.1986.10478342

### Spatial Models

Lindgren, F. (2010). Continuous domain spatial models in R-INLA. *Environmental and Ecological Statistics*, *18*, 165–183. https://doi.org/10.1007/s10651-009-0115-1

### Applied Network Analysis

Gulati, R., & Gargiulo, M. (1999). Where do interorganizational networks come from? *American Journal of Sociology*, *104*(5), 1439–1493. https://doi.org/10.1086/210179

### Dynamic Networks

Matias, C., & Miele, V. (2017). Statistical clustering of temporal dynamic networks. *Statistics and Computing*, *27*(4), 1065–1086.

### Software

Peixoto, T. P. (2014). The graph-tool Python library. https://graph-tool.skewed.de/

Peixoto, T. P. (2014). Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models. *Physical Review E*, *89*(1), 012804.

### Data

SocioPatterns Collaboration. (n.d.). *SocioPatterns*. http://www.sociopatterns.org/

Stehlé, J., Voirin, N., Barrat, A., Cattuto, C., Isella, L., Pinton, J.-F., Quaggiotto, M., Van den Broeck, W., Régis, C., Lina, B., & Vanhems, P. (2011). High-resolution measurements of face-to-face contact patterns in a primary school. *PLoS ONE*, *6*(8), e23176.

---

## Course Material

Marino, M. F. (2024–2025). *Network Data Analysis* [Lecture slides]. Master's Degree in Data Science and Statistical Learning (MD2SL), Dipartimento di Statistica, Informatica, Applicazioni (DiSIA), Università degli Studi di Firenze.

---

## Acknowledgments

We would like to thank **Prof. Maria Francesca Marino** for the excellent course material. The theoretical foundations presented in her lectures on network data analysis, stochastic block models, and community detection provided the essential framework for this implementation.

We also thank the developers and maintainers of the `graph-tool` library. Initially, we attempted to use **LaNet-vi** for network visualization, but despite extensive efforts, we were unable to get it to work properly (segmentation faults and compatibility issues). Thanks to `graph-tool`, we were able to complete this project entirely in Python — work that would otherwise have required R and its ecosystem of network analysis packages.

---

## Authors

**Orso Peruzzi** & **Giovanni Di Donato**

Master's students in Data Science and Statistical Learning (MD2SL)  
[IMT School for Advanced Studies Lucca](https://www.imtlucca.it/) & [University of Florence](https://www.unifi.it/), Italy

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project depends on the following open-source libraries:

| Library | License | Usage |
|---------|---------|-------|
| [graph-tool](https://graph-tool.skewed.de/) | LGPL v3 | Network analysis and SBM inference |
| [NumPy](https://numpy.org/) | BSD | Numerical computations |
| [SciPy](https://scipy.org/) | BSD | Scientific computing |
| [Matplotlib](https://matplotlib.org/) | PSF/BSD | Visualization |

These libraries retain their original licenses. Our MIT license applies only to the original code in this repository.

---

*Last updated: January 2026*
