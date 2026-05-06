# CS691LLMs

This repository contains the Python-based pipeline for creating datasets and training LLM-based NPC personas, specifically focusing on character resilience for models such as Gemma, Llama, Ministral, and Phi.

The Python components of this project can be found at: [https://github.com/joshuadahlunr/CS691LLMs](https://github.com/joshuadahlunr/CS691LLMs).
The Unity components can be found at: [https://github.com/AdrianDeli/AbbyBristow](https://github.com/AdrianDeli/AbbyBristow)

## Environment Management

This project uses two main Python environments: one for **dataset creation** and another for **training**. Both are managed with [uv](https://docs.astral.sh/uv/).

If you do not have `uv` installed, follow the [installation documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Dataset Creation

The raw data used to train the Holo models is located in `pairs.txt` and `fixed_conversations.txt`. The prompt structure is defined in `prompt_template.txt`.

To build the dataset:

1.  **Initialize the environment:** Run `uv sync`. Then, activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

2.  **Build the conversation database:**

    ```bash
    python create_conversation_db.py
    ```

3.  **Build the world databases:** There are five world databases used for training (`world1-5.txt`) and one for inference (`world6.txt`). Build them by running the following command for each (replacing both `#`'s with 1 through 5):

    ```bash
    python create_world_db.py db# world#.txt
    ```

4.  **Generate the training file:**

    ```bash
    python build_dataset.py > dataset.txt
    ```

    This produces the `dataset.txt` file used by the model trainers.

_Note: This process is also available in `combined.ipynb`, but this notebook was not used for the official user study and has undergone less rigorous testing._

## Model Training

To train a specific model, navigate into its respective folder (`gemma`, `llama`, `ministral`, `phi`, or `qwen`) and initialize the environment:

```bash
uv sync
source .venv/bin/activate
```

- **Venv Compatibility:** All training folders except `qwen` share the same environment settings.
- **Qwen Note:** At the time of writing, Qwen does not successfully convert to GGUF and was therefore excluded from the user study.
- **Execution:** Run the training script specific to the model (e.g., `train_gemma.py`). These scripts assume `dataset.txt` has already been created in the steps above. Each will produce a `holo_###_gguf` folder containing the GGUF model.

## Llama Conversion and Quantization

Llama is an exception and requires a manual process to create the GGUF model after training is complete.

### 1\. Conversion to GGUF

You must use `llama.cpp` for the conversion:

1.  Git clone [llama.cpp](https://github.com/ggml-org/llama.cpp).
2.  Navigate into the cloned folder and create a virtual environment:

```bash
    python -m venv venv
    source venv/bin/activate
    python -m pip install -r requirements.txt
```

3.  Run the conversion script:

```bash
    python convert_hf_to_gguf.py <path_to_CS691LLMs_folder>/llama/holo_llama3.1/
```

### 2\. Quantization

To quantize the model, you must build the `llama.cpp` repository. This requires `cmake` and a C++ build runtime.

1.  **Build the tools:**

```bash
    cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
    cmake --build build
```

2.  **Quantize (Q8 for User Study):**

```bash
    ./build/bin/llama-quantize <path_to_repo>/llama/holo_llama3.1/Holo_Llama3.1-8.0B-BF16.gguf <path_to_repo>/llama/holo_llama3.1/Holo_Llama3.1-8.0B-Q8.gguf Q8_0
```

3.  **Quantize for Unity (4-bit):**
    Replace `Q8_0` with `q4_k_m` in the command above.
