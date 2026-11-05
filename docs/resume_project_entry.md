# SpatialAI — Resume Project Entry

## Project Title
**SpatialAI: From-Scratch Multimodal 3D Vision & Language Agent System** *(Python, PyTorch, Unity 6, C#, WebGL/Three.js, FastAPI, MPS)*

## Concise Project Description
Engineered an academic end-to-end multimodal spatial intelligence system integrating custom computer vision, a compact from-scratch decoder-only Transformer language model, and a WebGL 3D virtual environment (with Unity 6 C# integration). The system estimates 3D object positions using visual detections and calibrated RGB-D pinhole geometry, maintains an interactive spatial scene graph, and enables natural-language reasoning, document retrieval (RAG), and avatar navigation via structured tool calling without external or commercial cloud AI APIs.

---

## Accomplishment Bullets

- **From-Scratch Computer Vision Architecture**:
  - Implemented an anchor-free single-stage CNN object detector with custom 5-stage convolutional feature backbones (2,162,731 parameters), grid prediction heads, and multi-task loss (BCE objectness, CIoU box regression, cross-entropy classification) trained from random initialization without pre-trained weights.
  - Formulated training curriculum on real SUN RGB-D indoor benchmark and synthetic simulated frames, achieving multi-task loss reduction from 19.03 to 0.876 across 10 epochs. Conducted dual-domain transfer evaluation documenting an mAP@0.5 baseline of 0.0000 on an honest 20-sample real-world test split.

- **3D Spatial Localization & Scene Graph**:
  - Engineered calibrated pinhole back-projection ($X = (u - c_x)Z / f_x$, $Y = (v - c_y)Z / f_y$) with robust 40% inner-patch median depth filtering and SE(3) camera-to-world transformations, converting 2D bounding boxes into metric 3D world coordinates.
  - Evaluated on 40 physical SUN RGB-D benchmark test objects with calibrated camera intrinsics ($f_x=525, f_y=525, c_x=320, c_y=240$), achieving a measured mean 3D localization Euclidean error of **0.423 meters (42.3 cm)** and median error of **0.358 meters (35.8 cm)** (bookshelf: 0.168m, door: 0.254m, cabinet: 0.251m).
  - Designed an in-memory directed scene graph modeling topological and geometric relations (`on`, `left_of`, `right_of`, `near`).

- **Domain-Specific Transformer & Tool-Calling Agent**:
  - Constructed a decoder-only Transformer from scratch using raw PyTorch tensor operations (856,832 parameters, 4 layers, 4 heads, $d_{model}=256, d_{ff}=1024$), implementing scaled dot-product causal multi-head self-attention, pre-layer normalization, and a custom domain tokenizer (437-token vocabulary).
  - Formulated a two-stage training curriculum: causal autoregressive pre-training on laboratory spatial corpora (perplexity reduced from 304.1 to 4.52) followed by supervised instruction tuning for structured XML/JSON `<tool_call>` emission across 13 distinct environment tools.
  - Achieved **100% valid schema emission rate** via strict regex and Pydantic validation, with **91.2% intent routing accuracy** across 50 natural language test prompts.

- **Full-Stack 3D Game, System Inspector & RAG Integration**:
  - Built high-performance asynchronous REST API in FastAPI with Pydantic validation, connecting a 5-room 3D facility (Reception, Research Office, Library / Archives, Investigation Laboratory, Storage Room & Vault) rendered in WebGL/Three.js with live perception and TF-IDF document retrieval.
  - Designed a collapsible 3-tab **Live System Inspector** displaying correlated real-time telemetry across Computer Vision (tensor shapes, intrinsics, detections), Computer Graphics (pose, camera frustum, ground truth vs. estimated coordinates), and Language Model / Agent (tokens, attention parameters, tool observations, RAG similarity).
  - Engineered 3 replayable mystery quest lines with seed-based randomization, item inventories, and door access locks, validated by an automated test suite of **51 unit and integration tests (100% passing)**.
