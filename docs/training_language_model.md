# Language Model Training Guide: Custom Transformer from Scratch

## 1. Educational Objectives (CS3235)
In accordance with **CS3235 Working with Large Language Models**, this module demonstrates building, pretraining, and instruction-tuning a compact decoder-only Transformer language model from random weights.

Pre-trained checkpoints (e.g. GPT-2, Llama, Hugging Face models) and commercial API wrappers are strictly forbidden in the core inference path.

## 2. Transformer Architecture from Scratch
- **Vocabulary Size**: 1024 tokens.
- **Embedding Dimension ($d_{model}$)**: 128.
- **Attention Heads ($n_{heads}$)**: 4 (head dimension $d_k = 32$).
- **Decoder Layers ($n_{layers}$)**: 4.
- **Feed-Forward Hidden Dimension ($d_{ff}$)**: 512 ($4 \times d_{model}$).
- **Positional Encoding**: Learnable or sinusoidal 1D positional embeddings.
- **Causal Masking**:
  An upper-triangular mask setting attention logits to $-\infty$ for $j > i$:
  $$\text{Mask}_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$
- **Pre-Layer Normalization**: Applied before self-attention and FFN with residual connections:
  $$x^{(1)} = x + \text{MHA}(\text{LayerNorm}(x))$$
  $$x^{(2)} = x^{(1)} + \text{FFN}(\text{LayerNorm}(x^{(1)}))$$

## 3. Two-Stage Training Curriculum

### Stage A: Autoregressive Domain Pretraining
- **Objective**: Teach the Transformer English syntactic patterns and spatial laboratory vocabulary.
- **Corpus**: Domain text consisting of object descriptions, spatial relations, equipment manuals, and room layouts.
- **Loss**: Standard causal cross-entropy loss on next-token prediction:
  $$\mathcal{L}_{LM} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

### Stage B: Supervised Instruction & Tool-Call Tuning
- **Objective**: Train the model to generate structured JSON tool calls given user queries and structured environmental context.
- **Prompt Format**:
  ```
  SYSTEM: You are an intelligent investigation drone.
  ROOM: Research Office
  VISIBLE OBJECTS:
  - table: [6.0, 0.8, -5.0]
  - chair: [6.0, 0.5, -4.0]
  SPATIAL RELATIONS:
  - chair is near table
  USER: Where is the chair?
  ASSISTANT:
  ```
- **Target Output**:
  `<thought>Retrieve the 3D metric coordinates of the chair in Research Office.</thought><tool_call>{"name": "locate_object", "arguments": {"object_name": "chair"}}</tool_call>`

---

## 4. Grounded Vision-Language Dataset (2,500 Dialogues)
Generated via `scripts/generate_lm_dataset.py` with zero synthetic mockery across 6 categories:
1. **Object Understanding** (400 examples): Object identification, counting, and metric coordinate retrieval.
2. **Spatial Reasoning** (500 examples): Euclidean distance calculation, relative direction (`left_of`, `right_of`), and proximity querying (`near`).
3. **Scene Understanding** (400 examples): Room identification, boundary description, and egress exit localization.
4. **Contextual & Negative Scenarios** (450 examples): Absent objects (`explain_uncertainty`), target reachability constraints, camera FOV margins, and multi-turn anaphora.
5. **Navigation & Agentic Interaction** (450 examples): Collision-free turn-by-turn routes (`navigate_to`), waypoint generation (`find_path`), and physical locks (`interact_with_object`).
6. **Investigation & RAG QA** (300 examples): Incident reports, passcodes, safety protocols, and anomalous artifact handling.

**Splits**:
- Training: 2,000 samples (`data/corpora/lm_train_dataset.jsonl`)
- Validation: 250 samples (`data/corpora/lm_val_dataset.jsonl`)
- Testing: 250 samples (`data/corpora/lm_test_dataset.jsonl`)

---

## 5. Multi-Dimensional 8-Metric Academic Benchmark
Evaluated on the 250 held-out test split via `scripts/evaluate_language_agent.py` (`results/metrics/agent_comprehensive_evaluation.json`):

| Metric # | Benchmark Dimension | Measured Result | Evaluation Scope |
| :--- | :--- | :--- | :--- |
| **1** | **Tool Selection Accuracy** | **36.00%** | Exact + semantic equivalence on 250 held-out test dialogues |
| **2** | **Object Identification Accuracy** | **26.32%** | Correct object name extraction and metric coordinate retrieval |
| **3** | **Spatial Reasoning Accuracy** | **43.86%** | Accurate distance, neighbor, and spatial direction identification |
| **4** | **Negative Observation Handling** | **35.71%** | Identifying absent objects and invoking `explain_uncertainty` |
| **5** | **Conversational Context Retention** | **84.62%** | Multi-turn anaphora resolution ("it", "that", "near it") |
| **6** | **RAG Answer Faithfulness** | **18.18%** | Document chunk citation matching ground-truth incident reports |
| **7** | **Hallucination Rate** | **0.00%** | Zero fabricated coordinates or impossible objects outside scene graph |
| **8** | **End-to-End Task Success Rate** | **99.20%** | Queries resulting in executable action without unhandled crashes |

---

## 6. Viva Demonstration Commands
1. Run full automated test suite:
   ```bash
   .venv/bin/python3 -m pytest tests/
   ```
2. Run 8-metric multi-dimensional evaluation:
   ```bash
   .venv/bin/python3 scripts/evaluate_language_agent.py
   ```
3. Interactive investigation companion chat:
   ```bash
   .venv/bin/python3 run.py --chat
   ```
