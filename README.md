# Temporal Network SBM

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![graph-tool](https://img.shields.io/badge/graph--tool-2.59-green.svg)](https://graph-tool.skewed.de/)

A comprehensive Python toolkit for analyzing **temporal (dynamic) networks** using **Stochastic Block Models (SBM)**.

This project was developed as an examination work for the course **Network Data Analysis** taught by Prof. **Maria Francesca Marino** at the University of Florence (Università degli Studi di Firenze), within the Master's program in **Data Science and Statistical Learning (MD2SL)**.

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

The Dynamic SBM analyzes how community structure evolves over time.

![Dynamic SBM Evolution](docs/dynamic_sbm_evolution.png)

The number of detected blocks varies between 1 and 9 across time windows:
- **Low activity periods** (night, early morning): fewer blocks detected (1-2)
- **High activity periods** (school hours): more blocks detected (5-9)

![Dynamic SBM Transitions](docs/dynamic_sbm_transitions.png)

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

This section provides the mathematical foundations underlying the implemented methods.

### 1. Network Centrality Measures

Centrality measures quantify the importance of nodes within a network. Different measures capture different aspects of "importance."

#### Degree Centrality

The simplest measure — counts the number of connections:

$$C_D(i) = \frac{k_i}{n-1}$$

where $k_i$ is the degree of node $i$ and $n$ is the total number of nodes. Normalized to [0, 1].

**Interpretation**: Nodes with high degree centrality are "popular" — they have many direct connections.

#### Betweenness Centrality

Measures how often a node lies on shortest paths between other nodes:

$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

where:
- $\sigma_{st}$ is the total number of shortest paths from $s$ to $t$
- $\sigma_{st}(i)$ is the number of those paths passing through $i$

**Interpretation**: High betweenness nodes are "brokers" — they control information flow between different parts of the network.

#### Closeness Centrality

Based on the average distance to all other nodes:

$$C_C(i) = \frac{n-1}{\sum_{j \neq i} d(i,j)}$$

where $d(i,j)$ is the shortest path distance between $i$ and $j$.

**Interpretation**: High closeness nodes can reach others quickly — they are "central" in a geographic sense.

#### Eigenvector Centrality

A node is important if it is connected to other important nodes:

$$C_E(i) = \frac{1}{\lambda} \sum_{j \in N(i)} C_E(j)$$

or in matrix form: $\mathbf{A} \mathbf{x} = \lambda \mathbf{x}$

where $\lambda$ is the largest eigenvalue of the adjacency matrix $\mathbf{A}$.

**Interpretation**: Measures influence — a node with few but highly central neighbors may be more central than a node with many peripheral neighbors.

#### Network Centralization (Freeman, 1979)

While centrality measures individual node importance, **centralization** measures how unequally centrality is distributed across the network:

$$C = \frac{\sum_{i=1}^{n} [C_{max} - C(i)]}{\max \sum_{i=1}^{n} [C_{max} - C(i)]}$$

- **C = 1**: Star network (one node dominates)
- **C = 0**: All nodes equally central (e.g., ring or complete graph)

---

### 2. Stochastic Block Model (SBM)

The Stochastic Block Model is a **generative probabilistic model** for networks with community structure.

#### Model Specification

1. Each node $i \in \{1, \ldots, n\}$ is assigned to a **block** $b_i \in \{1, \ldots, K\}$
2. The probability of an edge between nodes $i$ and $j$ depends **only** on their block memberships:

$$P(A_{ij} = 1 | b_i, b_j) = \theta_{b_i b_j}$$

where $\mathbf{\Theta}$ is a $K \times K$ matrix of connection probabilities.

#### Why SBM over Modularity?

| Aspect | SBM | Modularity (e.g., Louvain) |
|--------|-----|----------------------------|
| **Approach** | Generative model | Heuristic optimization |
| **Inference** | Probabilistic (Bayesian) | Greedy algorithm |
| **Structure detected** | Assortative AND disassortative | Only assortative |
| **Model selection** | Principled (MDL/BIC) | Resolution limit problem |
| **Uncertainty** | Provides posterior probabilities | Point estimate only |

The SBM can detect:
- **Assortative structure**: nodes connect within their block (communities)
- **Disassortative structure**: nodes connect between blocks (bipartite-like)
- **Core-periphery**: central blocks connect to all, peripheral blocks connect only to core

#### Inference via Minimum Description Length

We seek the block assignment $\mathbf{b}$ and parameters $\mathbf{\Theta}$ that best explain the observed network $\mathbf{A}$.

Using the **Minimum Description Length** principle:

$$\Sigma = -\log P(\mathbf{A} | \mathbf{\Theta}, \mathbf{b}) + \log P(\mathbf{\Theta}, \mathbf{b})$$

This balances:
- **Likelihood**: How well the model fits the data
- **Complexity**: Penalty for model complexity (many blocks = higher penalty)

The optimal number of blocks $K$ is chosen automatically by minimizing $\Sigma$.

**Implementation**: We use `graph-tool`'s highly optimized MCMC algorithms (Peixoto, 2014).

---

### 3. Dynamic Stochastic Block Model

In temporal networks, community structure may **evolve**:
- Nodes may change group membership
- New communities may form
- Existing communities may merge or split

#### Sliding Window Approach

Following Matias & Miele (2017), we implement a discrete-time approach:

1. **Partition time** into $T$ overlapping windows $[t_1, t_1+\Delta], [t_2, t_2+\Delta], \ldots$
2. For each window $w$, construct the **snapshot graph** $G_w$ from edges active during that window
3. Fit a static SBM to each $G_w$, obtaining block assignments $\mathbf{b}^{(w)}$
4. **Align labels** across consecutive windows (the label switching problem)
5. Compute **transition statistics**

#### The Label Switching Problem

Block labels are arbitrary — Block 1 in window $t$ may correspond to Block 3 in window $t+1$. We solve this using the **Hungarian algorithm**:

$$\pi^* = \arg\max_{\pi} \sum_{k=1}^{K} |B_k^{(t)} \cap B_{\pi(k)}^{(t+1)}|$$

where $\pi$ is a permutation of block labels.

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

- **Holland, P. W., & Leinhardt, S.** (1976).  
  *Local structure in social networks*.  
  **Sociological Methodology**, 7, 1–45.

- **Holland, P. W., & Leinhardt, S.** (1981).  
  *An exponential family of probability distributions for directed graphs*.  
  **Journal of the American Statistical Association**, 76(373), 33–50.

- **Holland, P. W., Laskey, K. B., & Leinhardt, S.** (1983).  
  *Stochastic blockmodels: First steps*.  
  **Social Networks**, 5(2), 109–137.

### Centrality

- **Freeman, L. C.** (1979).  
  *Centrality in social networks: Conceptual clarification*.  
  **Social Networks**, 1(3), 215–239.

### Latent Variable Models

- **Lazarsfeld, P. F., & Henry, N. W.** (1968).  
  *Latent structure analysis*.  
  Boston: Houghton Mifflin.

- **Hoff, P. D., Raftery, A. E., & Handcock, M. S.** (2002).  
  *Latent space approaches to social network analysis*.  
  **Journal of the American Statistical Association**, 97(460), 1090–1098.

### Stochastic Block Models

- **Snijders, T. A. B., & Nowicki, K.** (1997).  
  *Estimation and prediction for stochastic blockmodels for graphs with latent block structure*.  
  **Journal of Classification**, 14(1), 75–100.

- **Nowicki, K., & Snijders, T. A. B.** (2001).  
  *Estimation and prediction for stochastic blockstructures*.  
  **Journal of the American Statistical Association**, 96(455), 1077–1087.

- **Daudin, J. J., Picard, F., & Robin, S.** (2008).  
  *A mixture model for random graphs*.  
  **Statistics and Computing**, 18(2), 173–183.

- **Airoldi, E. M., Blei, D. M., Fienberg, S. E., & Xing, E. P.** (2008).  
  *Mixed membership stochastic blockmodels*.  
  **Journal of Machine Learning Research**, 9, 1981–2014.

### Dynamic Networks

- **Matias, C., & Miele, V.** (2017).  
  *Statistical clustering of temporal dynamic networks*.  
  **Statistical Computing**, 27(4), 1065–1086.

### Exponential Random Graph Models

- **Frank, O., & Strauss, D.** (1986).  
  *Markov graphs*.  
  **Journal of the American Statistical Association**, 81(395), 832–842.

### Applied Network Analysis

- **Gulati, R., & Gargiulo, M.** (1999).  
  *Where do interorganizational networks come from?*  
  **American Journal of Sociology**, 104(5), 1439–1493.

### Software

- **Peixoto, T. P.** (2014).  
  *The graph-tool Python library*.  
  [https://graph-tool.skewed.de/](https://graph-tool.skewed.de/)

- **Peixoto, T. P.** (2014).  
  *Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models*.  
  **Physical Review E**, 89(1), 012804.

### Data

- **SocioPatterns Project**  
  [http://www.sociopatterns.org/](http://www.sociopatterns.org/)

- **Stehlé, J., Voirin, N., Barrat, A., et al.** (2011).  
  *High-resolution measurements of face-to-face contact patterns in a primary school*.  
  **PLoS ONE**, 6(8), e23176.

---

## Course Material

This project is based on the lecture material from:

> **Marino, M. F.**  
> *Network Data Analysis*  
> Master's Degree in Data Science and Statistical Learning (MD2SL)  
> Dipartimento di Statistica, Informatica, Applicazioni (DiSIA)  
> Università degli Studi di Firenze  
> Academic Year 2024–2025

---

## Acknowledgments

I would like to thank **Prof. Maria Francesca Marino** for the excellent course material and guidance that made this project possible. The theoretical foundations presented in her lectures on network data analysis, stochastic block models, and community detection provided the essential framework for this implementation.

Thanks also to **Tiago Peixoto** for the exceptional `graph-tool` library, which makes efficient inference on large networks accessible to the research community.

---

## Author

**Orso Peruzzi**

Master's student in Data Science and Statistical Learning (MD2SL)  
University of Florence, Italy

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Last updated: January 2026*
