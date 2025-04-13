# CodeSight (Python AST & RAG 代码分析器)

本项目是一个使用 Python AST（抽象语法树）和 RAG（检索增强生成）技术的代码分析工具。它可以解析 Python 项目，构建代码元素（模块、类、函数）之间的依赖关系图，并利用大型语言模型（LLM）生成代码摘要、提取关系、生成项目报告以及回答用户关于代码库的问题。

## ✨ 主要功能

* **AST 解析**: 遍历 Python 代码，提取模块、类、函数定义及其元数据（文档字符串、行号等）。
* **调用关系提取**: 识别函数/方法内部的调用关系。
* **依赖图构建**: 基于解析信息和 LLM 提取的关系，构建代码元素间的依赖图 (使用 NetworkX)。
* **图可视化**: 在 Streamlit 应用中，使用 `streamlit-agraph` 可视化依赖关系图。
* **LLM 驱动的预处理**:
    * **代码摘要**: 为函数、类、模块生成自然语言摘要。
    * **关系提取**: (实验性) 尝试使用 LLM 识别更复杂的依赖关系或概念联系。
* **RAG 问答**:
    * **上下文检索**: 基于用户问题，使用 FAISS 向量索引检索相关的代码块。
    * **增强生成**: 结合检索到的上下文，使用 LLM 生成对用户问题的回答。
* **自动项目报告**: 使用 RAG 生成项目的技术概览报告。
* **灵活的 LLM 配置**: 支持通过 `.env` 文件配置使用不同的 LLM 提供商 (Ollama, DeepSeek) 和模型，可为不同任务（嵌入、生成、摘要等）指定不同的模型。
* **项目隔离**: 为每个分析的项目创建独立的目录，用于存储 RAG 索引 (FAISS) 和预处理缓存，避免项目间干扰。
* **忽略规则**: 支持通过配置文件 (`config/ignore.patterns`) 定义忽略特定文件或目录的规则，避免解析不必要的内容（如虚拟环境、测试数据等）。
* **Web 界面**: 使用 Streamlit 构建交互式 Web 应用，方便用户操作和查看结果。

## 📁 项目结构
```
code-sight
├── .env                 # 存储 API Keys 等敏感信息 (不提交到版本库)
├── .env.example         # .env 的模板文件 (提交到版本库)
├── .gitignore           # Git 忽略文件列表
├── config/              # 配置文件目录 (可选)
│   └── ignore.patterns  # 自定义忽略文件/目录规则
├── data/                # 存放 RAG 索引和缓存的根目录
│   └── project_hash_1/  # 特定项目的索引/缓存 (基于项目路径哈希)
│       ├── vector_index.faiss
│       ├── vector_metadata.pkl
│       └── preprocessed_chunks.pkl # 预处理结果缓存
│   └── project_hash_2/
│       └── ...
├── src/                 # 主要源代码目录
│   ├── __init__.py
│   ├── llm_clients/     # LLM 客户端抽象层
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ollama_client.py # 或 ollama.py (根据用户命名)
│   │   └── deepseek_client.py
│   │   └── factory.py     # LLM 客户端工厂
│   ├── utils/           # 工具函数
│   │   ├── init.py
│   │   ├── config_loader.py
│   │   └── hashing.py
│   ├── code_parser.py
│   ├── preprocessing.py
│   ├── rag_handler.py
│   └── graph_builder.py
├── tests/               # 测试代码目录 (本次未生成)
│   └── test_pipeline.py # 集成测试脚本 (待更新)
├── app.py               # Streamlit 应用入口
├── requirements.txt     # Python 依赖列表
└── README.md            # 本文件
```

## 🚀 安装与设置

**1. 环境准备:**

* 确保你安装了 Python (建议 3.8 或更高版本)。
* (可选但推荐) 创建并激活一个 Python 虚拟环境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

**2. 安装依赖:**

* 在项目根目录下，运行以下命令安装所有必需的库：
    ```bash
    pip install -r requirements.txt
    ```
    *注意：* `faiss-cpu` 可能需要 C++ 编译环境。如果安装失败，请查阅 [FAISS 文档](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) 获取特定平台的安装指南。通常，预编译的 wheel 可以简化安装。

**3. 配置 `.env` 文件:**

* 复制 `.env.example` 文件并重命名为 `.env`：
    ```bash
    cp .env.example .env
    ```
* 编辑 `.env` 文件，填入必要的信息：
    * **DeepSeek API Key**: 如果你计划使用 DeepSeek 模型进行任何任务（生成、摘要、关系提取），请在 `DEEPSEEK_API_KEY` 处填入你的有效 API 密钥。你可以从 [DeepSeek Platform](https://platform.deepseek.com/) 获取。
    * **Ollama 配置**: 如果你使用本地运行的 Ollama，请确保 `OLLAMA_BASE_URL` 正确（通常是 `http://localhost:11434`）。确保你在 Ollama 中已经拉取了 `.env` 文件中指定的 Ollama 模型（如 `nomic-embed-text`, `qwen2.5-coder:7b` 等）。
    * **模型标识符**: 根据你的需求和可用模型，配置以下变量，格式为 `"provider::model_name"`：
        * `EMBEDDING_MODEL_IDENTIFIER`: 用于生成文本嵌入的模型（**必须配置**，且不能是 `deepseek::...`）。
        * `GENERATION_MODEL_IDENTIFIER`: 用于 RAG 问答生成的模型。
        * `SUMMARIZATION_MODEL_IDENTIFIER`: 用于生成代码摘要的模型。
        * `RELATIONSHIP_MODEL_IDENTIFIER`: 用于提取代码关系的模型。
    * **嵌入维度 (`EMBEDDING_DIM`)**: **非常重要！** 必须设置为你选择的 `EMBEDDING_MODEL_IDENTIFIER` 所对应的嵌入向量维度（例如，`nomic-embed-text` 通常是 768）。错误的维度会导致 FAISS 索引失败。

**4. 配置忽略规则 (可选):**

* 如果需要忽略某些文件或目录（如虚拟环境、测试数据、日志文件等），可以在 `config/ignore.patterns` 文件中添加规则，语法类似 `.gitignore`。
* 确保 `.env` 文件中的 `IGNORE_PATTERNS_FILE` 指向这个文件（默认为 `./config/ignore.patterns`）。

## ⚙️ 配置详解

* **`.env` 文件**:
    * `OLLAMA_*`: Ollama 服务地址和默认使用的模型名称。
    * `DEEPSEEK_*`: DeepSeek API 密钥、服务地址和默认使用的模型名称。
    * `*_MODEL_IDENTIFIER`: 控制哪个 LLM 提供商和模型用于特定任务。这是实现灵活性的关键。
    * `EMBEDDING_DIM`: 必须与嵌入模型匹配。
    * `PREPROCESSING_MAX_WORKERS`: 控制预处理（调用 LLM）时的并行线程数。
    * `DATA_ROOT_DIR`: RAG 索引和缓存的根目录。
    * `LOG_LEVEL`: 应用的日志记录级别（DEBUG, INFO, WARNING, ERROR）。
    * `IGNORE_PATTERNS_FILE`: 指向忽略规则文件的路径。
* **`config/ignore.patterns`**:
    * 每行一个模式。
    * `#` 开头的行是注释。
    * 支持 `*` 等通配符（由 `fnmatch` 处理）。
    * 以 `/` 结尾的模式仅匹配目录。

## ▶️ 如何使用

1.  **启动 Streamlit 应用:**
    * 在项目根目录下，运行：
        ```bash
        streamlit run app.py
        ```
    * 应用将在你的浏览器中打开。

2.  **使用界面:**
    * **项目选择**: 在左侧边栏的文本框中输入你想要分析的 Python 项目的本地目录路径。
    * **开始分析**: 点击“分析项目”按钮。应用将执行以下步骤：
        * 解析代码 (应用忽略规则)。
        * 预处理代码块 (生成摘要和关系，使用配置的 LLM，结果会被缓存)。
        * 构建/加载 RAG 索引 (存储在 `data/项目哈希/` 目录下)。
        * 构建依赖关系图。
        * 生成初始的项目报告。
    * **查看依赖图**: 分析完成后，主要的依赖关系图会显示在左侧区域。你可以缩放、拖动、点击节点。
    * **查看项目报告**: 在右侧区域的“项目报告”选项卡中查看由 LLM 生成的报告。可以点击“重新生成报告”。
    * **互动分析**:
        * 点击左侧图中的节点（模块、类、函数），其详细信息（摘要、代码片段、提取的关系）会显示在右侧“互动分析”选项卡中。
        * **上下文管理**: 点击节点信息下方的“加入到上下文”按钮，将该节点添加到 RAG 问答的固定上下文中。当前上下文列表会显示在下方。可以“从上下文中移除”或“清空上下文”。
        * **RAG 问答**: 在“提问”文本框中输入关于代码的问题，然后点击“提问”按钮。
            * 如果选择了上下文节点，RAG 将优先使用这些节点的信息。
            * 如果没有选择上下文，RAG 会根据你的问题从索引中检索最相关的代码块作为上下文。
            * 最终，使用配置的生成模型结合上下文来回答你的问题。结果会显示在下方。

## 📦 依赖项

本项目依赖以下主要 Python 库 (详细列表请参见 `requirements.txt`):

* Streamlit & streamlit-agraph
* NetworkX
* FAISS (faiss-cpu)
* Ollama Python Client
* Requests
* python-dotenv
* Astor
* NumPy

