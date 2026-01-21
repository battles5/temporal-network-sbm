# Temporal Network SBM

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![graph-tool](https://img.shields.io/badge/graph--tool-2.59-green.svg)](https://graph-tool.skewed.de/)

A comprehensive Python toolkit for analysing **temporal (dynamic) networks** using **Stochastic Block Models (SBM)** and **hypergraph group extraction**.

This project was developed as an examination work for the course Network Data Analysis taught by Prof. Maria Francesca Marino at the University of Florence, within the Master's programme in Data Science and Statistical Learning MD2SL.

The toolkit implements rigorous statistical methods for network community detection, combining classical network analysis with modern inference-based approaches. It also includes an optional hypergraph analysis module that extracts group interactions via maximal clique enumeration, following the methodology of Iacopini et al. (2022). The complete pipeline transforms raw temporal edge data into publication-quality visualisations and comprehensive statistical reports.

![Network Animation Example](docs/network_animation.gif)

---

## Table of Contents

1. [Features](#features)
   - [Static Network Analysis](#static-network-analysis)
   - [Stochastic Block Model (SBM)](#stochastic-block-model-sbm)
   - [Dynamic Stochastic Block Model](#dynamic-stochastic-block-model)
   - [Hypergraph Group Extraction (Cliques)](#hypergraph-group-extraction-cliques)
   - [Visualisation Suite](#visualisation-suite)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Input Data Format](#input-data-format)
5. [Output](#output)
6. [Example: LyonSchool Dataset](#example-lyonschool-dataset)
7. [Configuration](#configuration)
8. [Theoretical Background](THEORY.md) ← *Separate document*
9. [References](#references)
10. [Acknowledgments](#acknowledgments)
11. [License](#license)

---

## Features

This toolkit provides a comprehensive suite of network analysis methods, structured around three principal pillars: static network analysis, stochastic block model inference, and temporal dynamics. Each component has been designed to offer rigorous statistical foundations whilst maintaining computational efficiency and ease of use.

### Static Network Analysis

The foundation of any network analysis begins with a thorough understanding of basic structural properties. The toolkit computes global statistics including the number of nodes, edges, network density, and the identification of connected components. Distance metrics such as network diameter and average path length provide insight into the overall navigability of the graph. Degree analysis encompasses the full distribution of node degrees alongside summary statistics including mean, standard deviation, minimum, maximum, and median values.

Clustering behaviour is assessed through both the local clustering coefficient and the global transitivity measure, offering complementary perspectives on triadic closure within the network. The toolkit further implements a comprehensive suite of centrality measures: degree centrality captures immediate connectivity, betweenness centrality quantifies the extent to which nodes lie on shortest paths between others, closeness centrality measures the average distance from each node to all others, and eigenvector centrality identifies nodes connected to other well-connected nodes. Finally, network centralisation is computed using Freeman's index, which quantifies the extent to which the network structure is dominated by a single node or small group of nodes.

### Stochastic Block Model (SBM)

Moving beyond heuristic community detection methods, this toolkit employs principled Bayesian inference for network partitioning. The optimal number of blocks is determined automatically through the **Minimum Description Length (MDL)** criterion, which balances model complexity against goodness of fit. Nodes are assigned to blocks via a hard partition derived from the maximum a posteriori (MAP) estimate obtained through Markov Chain Monte Carlo (MCMC) inference.

The inter-block connection probability matrix Π characterises the propensity for edges to form between and within blocks, whilst internal block density analysis reveals the cohesiveness of each community. For comparison with alternative model selection criteria, the **Integrated Classification Likelihood (ICL)** is also computed.

### Dynamic Stochastic Block Model

For temporal networks, the toolkit implements a sliding-window approach inspired by the `dynsbm` methodology. Temporal data are partitioned into overlapping time windows, with an independent SBM fitted to each snapshot. A critical challenge in this setting is the correspondence problem: block labels are arbitrary within each window, making direct comparison across time points problematic. This is addressed through **label alignment** using the Hungarian algorithm, which finds the optimal permutation of labels to maximise consistency between consecutive windows.

The aligned block assignments enable the computation of a **transition probability matrix** $P(b_{t+1} | b_t)$, which characterises how nodes move between communities over time. Block stability metrics quantify the persistence of community structure, whilst the identification of **mobile nodes**—those individuals who frequently change block membership—reveals the dynamic core of the network.

### Hypergraph Group Extraction (Cliques)

Many real-world social interactions involve more than two individuals simultaneously. Whilst temporal edge lists record only pairwise contacts, it is possible to approximate **higher-order group interactions** by extracting cliques from aggregated time windows, following the methodology of Iacopini et al. (2022).

The underlying rationale is straightforward: within a given time window, if individuals A, B, and C all interact pairwise (A↔B, B↔C, A↔C), they have likely participated in a group interaction. This pattern corresponds to a clique—that is, a complete subgraph—in the contact network. The toolkit extracts **maximal cliques** from each time-window snapshot, where each clique of size k represents a k-person group interaction, or equivalently, a hyperedge of order k. From these extractions, the distribution of group sizes across all windows is computed, and the temporal evolution of group sizes is tracked throughout the observation period.

This analysis is optional and may be enabled with the `--hypergraph` flag.

### Visualisation Suite

All analyses are accompanied by publication-ready visualisations designed to communicate results effectively. For degree analysis, both linear and log-log scale distributions are produced. Centrality measures are presented through comparison scatter plots alongside network graphs where node sizes reflect centrality values. The SBM results are visualised through community structure diagrams and block connection probability heatmaps.

Temporal aspects of the data are captured through activity timelines, whilst the dynamic SBM results are presented via transition heatmaps and block evolution plots. When hypergraph analysis is enabled, additional figures display the group size distribution and the temporal trajectory of group sizes (median and interquartile range per window). An optional animated network evolution visualisation is also available, with configurable resolution settings.

Further details regarding output filenames may be found in the [Output](#output) section below.

---

## Installation

### System Requirements

- **Python 3.8+**
- **Linux or WSL** (Windows Subsystem for Linux) — required for graph-tool
- 4GB RAM recommended for large networks

### Step 1: Install graph-tool

The core of this toolkit relies on `graph-tool`, a highly efficient C++ library for network analysis. It cannot be installed via pip.

**Conda (recommended — works reliably on Linux, macOS, and WSL)**
```bash
conda create -n netsbm python=3.10
conda activate netsbm
conda install -c conda-forge graph-tool
```

**Alternative: Ubuntu/Debian via apt (may require additional configuration)**
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install python3-graph-tool
```
> ⚠️ The `apt` method may fail on some Ubuntu versions. If you encounter issues, use Conda instead.

**Docker (containerized)**
```bash
docker pull tiagopeixoto/graph-tool
docker run -it tiagopeixoto/graph-tool python3
```

### Step 2: Install Python dependencies

With your conda environment activated:

```bash
conda activate netsbm
pip install -r requirements.txt
```

`requirements.txt` includes NumPy, SciPy, Matplotlib, pandas, PyYAML, and NetworkX (used for clique enumeration in hypergraph analysis).

> ⚠️ **Important**: Always install packages inside the activated conda environment. Do not use `sudo` with pip or conda.

### Step 3: Clone this repository

```bash
git clone https://github.com/battles5/temporal-network-sbm.git
cd temporal-network-sbm
```

### Windows Users (WSL)

This toolkit requires **WSL** (Windows Subsystem for Linux):

1. Open PowerShell as Administrator and run: `wsl --install`
2. Install Ubuntu from Microsoft Store (22.04 or 24.04)
3. Inside WSL, install Miniconda and then graph-tool via conda-forge:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   conda create -n netsbm python=3.10
   conda activate netsbm
   conda install -c conda-forge graph-tool
   ```
4. Run the toolkit from within WSL

---

## Quick Start

### Dataset

> **Note**: This repository does not include datasets. Download a temporal network dataset before running the analysis.

**Recommended: LyonSchool (primary school face-to-face contacts)**
- [SocioPatterns: Primary School Dataset](http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/)
- [Netzschleuder: sp_primary_school](https://networks.skewed.de/net/sp_primary_school)

Other compatible datasets: SFHH, InVS, Thiers, Hospital (all from [SocioPatterns](http://www.sociopatterns.org/datasets/)).

**Setup**: Create the `data/` folder and place your downloaded dataset there:

```bash
mkdir -p data
# Place the downloaded dataset here, e.g.:
# data/tij_LyonSchool.dat
```

The toolkit expects a simple edge list format:
```
timestamp  node1  node2
0          1      2
0          2      3
20         1      3
...
```

### Basic Usage

```bash
python main.py --input <data_file> --output <output_dir>
```

### Example Commands

```bash
# Run analysis on LyonSchool (download from SocioPatterns first)
python main.py --input data/tij_LyonSchool.dat --output output/

# With custom configuration
python main.py --input data/mydata.dat --output output/ --config my_config.yaml

# Generate network animation (resource-intensive)
python main.py --input data/mydata.dat --output output/ --animate

# Skip dynamic SBM for faster execution
python main.py --input data/mydata.dat --output output/ --no-dynamic-sbm

# Enable hypergraph group extraction (clique analysis)
python main.py --input data/mydata.dat --output output/ --hypergraph

# Hypergraph with custom group size limits
python main.py --input data/mydata.dat --output output/ --hypergraph --min-group-size 4 --max-group-size 15
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--input` | `-i` | Path to input data file (required) |
| `--output` | `-o` | Path to output directory (required) |
| `--config` | `-c` | Path to YAML configuration file |
| `--animate` | | Generate network animation (MP4/GIF) |
| `--no-dynamic-sbm` | | Skip dynamic SBM analysis |
| `--hypergraph` | | Enable hypergraph group extraction via cliques |
| `--min-group-size` | | Minimum clique size (default: 3) |
| `--max-group-size` | | Maximum clique size (default: 20) |
| `--max-cliques-per-window` | | Safety limit for clique enumeration |

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
│   ├── group_size_distribution.png  (if --hypergraph)
│   ├── group_size_over_time.png     (if --hypergraph)
│   └── network_animation.mp4        (if --animate)
├── metrics.csv
├── top_nodes.csv
├── sbm_results.csv
├── dynamic_sbm_windows.csv
├── dynamic_sbm_transitions.csv
├── dynamic_sbm_stability.csv
├── hypergraph_groups.csv            (if --hypergraph)
├── group_size_distribution.csv      (if --hypergraph)
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
| `hypergraph_groups.csv` | All extracted groups: window_id, group_id, group_size, node_ids |
| `group_size_distribution.csv` | Distribution of group sizes: size, count, proportion |
| `summary.txt` | Human-readable comprehensive report |

---

## Example: LyonSchool Dataset

To demonstrate the toolkit's capabilities, we analyse the **LyonSchool** dataset from the [SocioPatterns project](http://www.sociopatterns.org/).

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
# Standard analysis (SBM + Dynamic SBM)
python main.py \
  --input data/tij_LyonSchool.dat \
  --output output/

# With hypergraph group extraction
python main.py \
  --input data/tij_LyonSchool.dat \
  --output output/ \
  --hypergraph
```

### Results Summary

Below are the complete results obtained from our analysis pipeline.

> **Reproducibility note**: These values are from one run with default settings (window size 300s, step 60s). Results may vary slightly depending on MCMC initialization and parameter choices.

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

Network visualisation with node sizes proportional to degree centrality. The graph reveals a relatively dense core with some peripheral nodes.

---

#### Network Centralization (Freeman's Index)

| Centralization Measure | Value | Interpretation |
|------------------------|-------|----------------|
| **Degree** | 0.2731 | Moderately centralized |
| **Betweenness** | 0.0103 | Low centralization |
| **Closeness** | 0.1128 | Slightly centralized |

The network is **slightly centralized** overall. The low betweenness centralization (~1%) indicates no single node dominates shortest paths, which is expected in a dense network with short diameter. The moderate degree centralization indicates some individuals are more socially active than others.

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

> **Note**: Node IDs (1551, 1780, etc.) are the original participant identifiers from the SocioPatterns dataset, not sequential indices. The 242 unique participants have IDs ranging from 1426 to 1922.

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

##### Attribute Assortativity (True Assortativity Coefficient)

When **external node attributes** are available (e.g., school class labels), the toolkit computes the **true attribute assortativity coefficient** following the course definition:

$$r = \frac{\text{Tr}(e) - \sum_i a_i^2}{1 - \sum_i a_i^2} = \frac{Q}{Q_{max}}$$

where:
- $e_{ij}$ = fraction of edges connecting class $i$ to class $j$
- $a_i = \sum_j e_{ij}$ = fraction of edge endpoints in class $i$
- $\text{Tr}(e) = \sum_i e_{ii}$ = fraction of edges within same class

| Attribute Assortativity | Value | Description |
|-------------------------|-------|-------------|
| **Number of classes** | 11 | 1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B, 5A, 5B, Teachers |
| **Modularity Q** | 0.2111 | Same-class edges above random |
| **Modularity Q_max** | 0.9031 | Maximum possible |
| **Assortativity r = Q/Q_max** | **0.2338** | Moderate assortative mixing |

**Interpretation**: $r = 0.2338$ indicates **moderate assortative mixing** — students preferentially interact with classmates, but there is significant inter-class interaction during breaks and lunch.

##### Partition Modularity (SBM Blocks)

The toolkit also reports **partition modularity** based on the inferred SBM blocks (not external attributes):

| Partition Metric | Value |
|------------------|-------|
| Partition Modularity Q | 0.1247 |
| Partition Q_max | 0.9306 |
| Partition Assortativity | 0.134 |

> **Important distinction**: Partition modularity measures the quality of the SBM partition, while attribute assortativity measures true homophily based on known node labels. These are different concepts!

##### Block Sizes and Densities

| Block | Size | Internal Density |
|-------|------|------------------|
| Block 0 | 8 | 0.857 |
| Block 1 | 11 | 1.000 |
| Block 2 | 11 | 0.946 |
| Block 3 | 15 | 1.000 |
| Block 4 | 13 | 1.000 |
| Block 5 | 16 | 0.983 |
| Block 6 | 23 | 0.968 |
| Block 7 | 16 | 0.950 |
| Block 8 | 11 | 1.000 |
| Block 9 | 16 | 0.992 |
| Block 10 | 16 | 0.992 |
| Block 11 | 20 | 0.805 |
| Block 12 | 8 | 1.000 |
| Block 13 | 11 | 0.982 |
| Block 14 | 8 | 1.000 |
| Block 15 | 17 | 0.993 |
| Block 16 | 6 | 1.000 |
| Block 17 | 16 | 0.983 |

The SBM identifies **18 blocks**, which is more than the 11 actual school classes. This finer granularity suggests the model detects **sub-groups within classes** (e.g., friend clusters, seating arrangements) or distinguishes students with different interaction patterns. The internal density close to 1.0 indicates that students within the same block interact with almost everyone else in their block.

> **Note on density = 1.0**: An internal density of 1.0 means the block is a *complete subgraph* (clique) — every pair of nodes within the block has an edge. This is expected in school classes where all students had at least one contact during the two-day observation period.

![Block Connection Matrix](docs/sbm_block_matrix.png)

**Figure: Connection Probability Matrix Π̂.** Each cell π̂_qℓ represents the estimated probability of an edge between a node in block q and a node in block ℓ.

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

The Dynamic SBM analyses how community structure evolves over time by fitting independent SBMs to each time window and aligning labels across consecutive windows. It is important to note that this analysis is fundamentally different from the static SBM described above: whilst the static SBM analyses the **entire aggregated network** (all interactions over two days, yielding 18 blocks corresponding to school classes), the Dynamic SBM fits a **separate model to each temporal window**. Since each window captures only a snapshot of activity (39 minutes), fewer nodes are active and consequently fewer blocks are detected per window (typically between 3 and 10, with a configured maximum of 10).

![Dynamic SBM Evolution](docs/dynamic_sbm_evolution.png)

**Figure: Block Size Heatmap.** Number of nodes in each block (rows) across time windows (columns). Darker cells indicate larger blocks; white cells indicate absent blocks.

**Interpretation notes:**
- Each row represents a block label (0, 1, 2, ...) as assigned after Hungarian algorithm alignment
- The number of active blocks varies across time windows depending on network activity
- **Low activity periods** (windows ~17–27, corresponding to night): 1–3 blocks detected
- **High activity periods** (school hours): up to 10 blocks detected

> **Caveat on label alignment**: The Hungarian algorithm aligns labels locally between consecutive windows $(t, t+1)$, which can cause "label drift" over longer periods. A block that appears as "Block 0" at $t=0$ may be relabeled as "Block 3" by $t=20$ due to accumulated misalignments. The transition matrix and heatmap should be interpreted with this limitation in mind.

![Dynamic SBM Transitions](docs/dynamic_sbm_transitions.png)

**Figure: Transition Probability Matrix P̂.** Each cell p̂_rs = P(b_{t+1}=s | b_t=r) represents the estimated probability that a node moves from block r at time t to block s at time t+1.

The transition matrix shows block-to-block movement probabilities. Key observations:
- **Diagonal dominance**: nodes tend to stay in their blocks (stable class membership)
- **Some off-diagonal flow**: students occasionally interact with other classes
- **Block stability ranges from 35% to 80%** depending on the class

| Windows Analysed | 50 |
|------------------|-----|
| Block stability range | 0.35 – 0.80 |
| Nodes that changed blocks | 242 (all) |

All 242 nodes changed blocks at least once, reflecting the natural dynamics of school life — students temporarily join different groups during breaks, lunch, or cross-class activities.

---

#### Hypergraph Group Extraction (Optional)

When running with `--hypergraph`, the toolkit extracts group interactions by identifying maximal cliques in each time window.

```bash
python main.py --input data/tij_LyonSchool.dat --output output/ --hypergraph
```

![Group Size Distribution](docs/group_size_distribution.png)

**Figure: Distribution of group sizes** extracted via clique enumeration. Left: linear scale; Right: log scale.

| Hypergraph Metric | Value |
|-------------------|-------|
| **Total groups extracted** | 73,950 |
| **Most common group size** | 3 (triangles) |
| **Largest groups observed** | 10–15 individuals |
| **Windows with groups** | 1,028 / 1,039 |
| **Analysis time** | ~3 seconds |

The group size distribution typically follows a **power-law-like decay**: many small groups (triads, tetrads) and progressively fewer large groups. This is consistent with the observation that full-class interactions are rare, while small-group conversations are frequent.

![Group Size Over Time](docs/group_size_over_time.png)

**Figure: Median group size per time window** with interquartile range (IQR). The pattern mirrors overall activity — larger groups form during peak school hours.

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
  max_windows: 50            # Number of windows to analyse
  max_blocks: 10             # Max blocks per window

# Visualisation settings
visualisation:
  dpi: 300
  format: "png"

# Animation settings
animation:
  enabled: false
  max_frames: 100
  fps: 10
  resolution: [1920, 1080]   # HD, use [3840, 2160] for 4K

# Hypergraph group extraction (clique-based)
# See: Iacopini et al. (2022) https://doi.org/10.1038/s42005-022-00845-y
hypergraph:
  enabled: false             # Enable with --hypergraph flag
  min_group_size: 3          # Minimum clique size (3 = triangles+)
  max_group_size: 20         # Safety limit for large cliques
  max_cliques_per_window: 10000  # Safety limit per window
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Path to input temporal edge list (required) |
| `--output`, `-o` | Path to output directory (required) |
| `--config`, `-c` | Path to configuration YAML file |
| `--animate` | Generate network animation (slow) |
| `--no-dynamic-sbm` | Skip dynamic SBM analysis |
| `--hypergraph` | Enable hypergraph group extraction via cliques |
| `--min-group-size` | Minimum group/clique size (default: 3) |
| `--max-group-size` | Maximum group/clique size (default: 20) |
| `--max-cliques-per-window` | Safety limit for clique enumeration (default: 10000) |

---

## Theoretical Background

📚 **The complete theoretical background has been moved to a separate document for clarity.**

➡️ **See [THEORY.md](THEORY.md)** for the full mathematical foundations, including:
- Basic notation and adjacency matrices
- Global statistics: density, reciprocity, transitivity
- Node centrality and network centralisation (Freeman's index)
- Stochastic Block Model (SBM): latent variables, VEM, ICL
- SBM vs modularity-based methods
- Assortativity coefficient (attribute vs partition)
- Temporal extension (Dynamic SBM)
- Hypergraph group extraction via cliques

All notation follows the *Network Data Analysis* course (M.F. Marino, University of Florence).

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

### Dynamic Networks

Matias, C., & Miele, V. (2017). Statistical clustering of temporal dynamic networks. *Statistics and Computing*, *27*(4), 1065–1086.

### Higher-Order Interactions and Hypergraphs

Iacopini, I., Petri, G., Baronchelli, A., & Barrat, A. (2022). Group interactions modulate critical mass dynamics in social convention. *Communications Physics*, *5*, 64. https://doi.org/10.1038/s42005-022-00845-y

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

We also thank the developers and maintainers of the `graph-tool` library. Initially, we attempted to use **LaNet-vi** for network visualisation, but despite extensive efforts, we were unable to get it to work properly (segmentation faults and compatibility issues). Thanks to `graph-tool`, we were able to complete this project entirely in Python — work that would otherwise have required R and its ecosystem of network analysis packages.

---

## Authors

**Orso Peruzzi** & **Giovanni Di Donato**

Master's students in Data Science and Statistical Learning (MD2SL)  
[IMT School for Advanced Studies Lucca](https://www.imtlucca.it/) & [University of Florence](https://www.unifi.it/), Italy

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

For the full Python dependency list, see `requirements.txt`.

| Library | License | Usage |
|---------|---------|-------|
| [graph-tool](https://graph-tool.skewed.de/) | LGPL v3 | Network analysis and SBM inference |
| [NumPy](https://numpy.org/) | BSD | Numerical computations |
| [SciPy](https://scipy.org/) | BSD | Scientific computing |
| [Matplotlib](https://matplotlib.org/) | PSF/BSD | Visualisation |
| [pandas](https://pandas.pydata.org/) | BSD | Data manipulation |
| [PyYAML](https://pyyaml.org/) | MIT | Configuration parsing |
| [NetworkX](https://networkx.org/) | BSD | Clique enumeration (hypergraph) |

These libraries retain their original licenses. Our MIT license applies only to the original code in this repository.

---

*Last updated: January 2026*
