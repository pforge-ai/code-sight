# app.py
# -*- coding: utf-8 -*-

"""
(V2 Refactor) Streamlit 主应用程序，整合了重构后的模块。
- 使用 src/ 目录下的代码解析、预处理、图构建和 RAG 模块。
- RAG Handler 负责处理项目特定的索引/缓存和 LLM 交互。
- 支持通过配置选择不同的 LLM 提供商和模型。
"""

import streamlit as st
import os
import logging
import networkx as nx
import pickle
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Dict, List, Any, Optional, Tuple

# --- Import Refactored Modules ---
# Ensure src is importable (e.g., by running streamlit run app.py from the project root)
try:
    from src.utils import config_loader as cfg # Load config early to set logging
    from src import code_parser
    from src import preprocessing
    from src import graph_builder
    from src.rag_handler import RAGHandler # Import the class
except ImportError as e:
     st.error(f"Failed to import necessary modules from 'src/'. Make sure you run streamlit from the project root directory. Error: {e}")
     st.stop() # Stop execution if imports fail


# --- Basic Setup ---
logger = logging.getLogger(__name__) # Get logger configured by config_loader

# Define filename for preprocessed data cache within project data dir
PREPROCESSED_DATA_FILENAME = "preprocessed_chunks.pkl"

# --- Helper Function for Agraph Conversion (Mostly Unchanged) ---
def convert_nx_to_agraph(graph: nx.DiGraph) -> Tuple[List[Node], List[Edge]]:
    """将 NetworkX 图转换为 streamlit-agraph 需要的节点和边列表。"""
    agraph_nodes = []
    agraph_edges = []

    if not graph:
        return agraph_nodes, agraph_edges

    default_node_color = "#CCCCCC"
    node_styles = {
        'function': {'color': '#74A9D8', 'shape': 'dot', 'size': 15},
        'class': {'color': '#F0A367', 'shape': 'box', 'size': 20},
        'module': {'color': '#90EE90', 'shape': 'database', 'size': 25},
        'unknown': {'color': default_node_color, 'shape': 'ellipse', 'size': 10}
    }
    default_edge_color = "#CCCCCC"
    edge_styles = {
        'CONTAINS': {'color': '#90EE90', 'dashes': True, 'label': ''},
        'CALLS': {'color': '#ADD8E6', 'dashes': False},
        'DEPENDS_ON': {'color': '#FFA07A', 'dashes': True},
        'IMPLEMENTS': {'color': '#FFB6C1', 'dashes': False}, # Example for new type
        'RELATES_TO': {'color': '#D3D3D3', 'dashes': True}, # Example for generic relation
        # Add more styles based on relationship types extracted by LLM
    }


    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        style = node_styles.get(node_type, node_styles['unknown'])

        # Ensure label is a string, use short_name if available
        node_label = str(node_data.get('short_name', node_data.get('label', node_id)))

        agraph_nodes.append(Node(id=node_id,
                                 label=node_label,
                                 title=node_data.get('title', ''), # Hover text comes from graph builder
                                 size=style['size'],
                                 shape=style['shape'],
                                 color=style['color']
                                 ))

    for u, v, edge_data in graph.edges(data=True):
        edge_type = edge_data.get('type', 'UNKNOWN')
        style = edge_styles.get(edge_type, {'color': default_edge_color, 'dashes': False, 'label': edge_type})

        # Ensure label is a string
        edge_label = str(style.get('label', edge_data.get('label', ''))) # Use style label first

        agraph_edges.append(Edge(source=u,
                                 target=v,
                                 label=edge_label,
                                 type="CURVE_SMOOTH", # Edge shape
                                 color=style['color'],
                                 dashes=style['dashes']
                                 ))

    return agraph_nodes, agraph_edges


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Python 代码分析器 (Refactored)")
st.title("🐍 Python AST & RAG 代码分析器 (Refactored)")

# --- Session State Initialization ---
# Use more robust initialization
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.project_dir = "."
    st.session_state.analysis_done = False
    st.session_state.analysis_error = None
    st.session_state.all_nodes_info = None
    st.session_state.all_imports_info = None
    st.session_state.preprocessed_chunks = None
    st.session_state.graph = None
    st.session_state.agraph_nodes = []
    st.session_state.agraph_edges = []
    st.session_state.rag_handler_instance = None
    st.session_state.selected_node_id = None
    st.session_state.selected_context_ids = [] # List to store IDs added to context
    st.session_state.project_report = ""
    st.session_state.project_report_error = None
    st.session_state.last_rag_question = ""
    st.session_state.last_rag_answer = ""
    logger.info("Streamlit session state initialized.")


# --- Sidebar ---
st.sidebar.header("项目选择 (Project Selection)")
project_dir_input = st.sidebar.text_input("输入 Python 项目目录路径 (Enter Python project directory path):", st.session_state.project_dir)
analyze_button = st.sidebar.button("分析项目 (Analyze Project)", key="analyze_btn")

# Display analysis status or errors in sidebar
if st.session_state.analysis_done and not st.session_state.analysis_error:
     st.sidebar.success(f"分析完成: {os.path.basename(st.session_state.project_dir)}")
elif st.session_state.analysis_error:
     st.sidebar.error(f"分析失败: {st.session_state.analysis_error}")


# --- Main Analysis Logic ---
if analyze_button:
    abs_project_dir = os.path.abspath(project_dir_input)
    if os.path.isdir(abs_project_dir):
        st.session_state.project_dir = abs_project_dir # Store absolute path
        # Reset state for new analysis
        st.session_state.analysis_done = False
        st.session_state.analysis_error = None
        st.session_state.all_nodes_info = None
        st.session_state.all_imports_info = None
        st.session_state.preprocessed_chunks = None
        st.session_state.graph = None
        st.session_state.agraph_nodes = []
        st.session_state.agraph_edges = []
        st.session_state.rag_handler_instance = None # Reset RAG handler
        st.session_state.selected_node_id = None
        st.session_state.selected_context_ids = []
        st.session_state.project_report = ""
        st.session_state.project_report_error = None
        st.session_state.last_rag_question = ""
        st.session_state.last_rag_answer = ""

        st.sidebar.info(f"开始分析 (Starting analysis): {st.session_state.project_dir}")
        logger.info(f"Starting analysis for project: {st.session_state.project_dir}")

        # --- Execute Full Pipeline ---
        try:
            # 0. Initialize RAG Handler (needs project path for data isolation)
            # Do this early as other steps might depend on its data dir
            with st.spinner("步骤 0/5: 初始化 RAG 系统... (Initializing RAG system...)"):
                 try:
                      st.session_state.rag_handler_instance = RAGHandler(project_path=st.session_state.project_dir)
                      project_data_dir = st.session_state.rag_handler_instance.project_data_dir # Get project specific dir
                      preprocessed_data_path = os.path.join(project_data_dir, PREPROCESSED_DATA_FILENAME)
                      logger.info("RAG Handler initialized.")
                 except ValueError as rag_init_err:
                      st.error(f"无法初始化 RAG 系统: {rag_init_err}")
                      st.session_state.analysis_error = f"RAG Init failed: {rag_init_err}"
                      st.stop() # Stop if RAG handler fails (e.g., missing keys)


            # 1. Parsing
            with st.spinner("步骤 1/5: 正在解析 Python 文件... (Parsing Python files...)"):
                st.session_state.all_nodes_info, st.session_state.all_imports_info = code_parser.parse_project(st.session_state.project_dir)
                if not st.session_state.all_nodes_info and not st.session_state.all_imports_info:
                    st.warning("解析项目时未找到任何有效的 Python 节点或导入。请检查项目路径和忽略规则。(No nodes/imports found. Check path and ignore rules.)")
                    # Allow continuing, maybe project is empty or fully ignored
                logger.info(f"解析完成。找到 {len(st.session_state.all_nodes_info)} 个文件包含节点。 (Parsing complete.)")


            # 2. Preprocessing (with caching per project)
            with st.spinner("步骤 2/5: 正在预处理代码块 (可能需要几分钟)... (Preprocessing code chunks...)"):
                 # Try loading cached preprocessed data for this project
                 loaded_from_cache = False
                 if os.path.exists(preprocessed_data_path):
                      logger.info(f"尝试从缓存加载预处理数据: {preprocessed_data_path}")
                      try:
                          with open(preprocessed_data_path, 'rb') as f:
                              st.session_state.preprocessed_chunks = pickle.load(f)
                          logger.info(f"成功加载 {len(st.session_state.preprocessed_chunks)} 个预处理块。(Successfully loaded preprocessed data.)")
                          loaded_from_cache = True
                      except Exception as load_err:
                           logger.warning(f"加载预处理缓存失败: {load_err}。将重新生成。(Failed to load cache, regenerating.)")
                           st.session_state.preprocessed_chunks = None

                 if not loaded_from_cache:
                      logger.info("执行 preprocessing.preprocess_project...")
                      st.session_state.preprocessed_chunks = preprocessing.preprocess_project(
                          st.session_state.all_nodes_info or {}, # Pass empty dict if None
                          st.session_state.all_imports_info or {}
                      )
                      if not st.session_state.preprocessed_chunks:
                           st.warning("预处理步骤未生成任何代码块。(Preprocessing generated no chunks.)")
                           # Allow continuing, graph/RAG might be empty
                      else:
                           # Save the result to project-specific cache
                           try:
                                with open(preprocessed_data_path, 'wb') as f:
                                     pickle.dump(st.session_state.preprocessed_chunks, f)
                                logger.info(f"预处理结果已保存到缓存: {preprocessed_data_path} (Preprocessed data saved to cache.)")
                           except Exception as save_err:
                                logger.error(f"保存预处理结果失败: {save_err} (Failed to save preprocessed data.)")
                 logger.info("预处理完成。(Preprocessing complete.)")


            # 3. RAG Indexing (using the initialized handler)
            with st.spinner("步骤 3/5: 正在构建/加载 RAG 索引... (Building/Loading RAG index...)"):
                 if st.session_state.rag_handler_instance and st.session_state.preprocessed_chunks:
                      # Build index (force_rebuild=True ensures consistency with fresh analysis)
                      # The handler itself tries loading first if files exist, but rebuild ensures freshness for the UI run.
                      # Consider making force_rebuild optional via UI? For now, rebuild.
                      st.session_state.rag_handler_instance.build_index(
                          st.session_state.preprocessed_chunks,
                          force_rebuild=True
                      )
                      if st.session_state.rag_handler_instance.index is None:
                           st.warning("构建 RAG 索引失败或索引为空。(RAG index build failed or index is empty.)")
                 elif not st.session_state.preprocessed_chunks:
                      st.info("没有预处理数据可用于构建 RAG 索引。(No preprocessed data for RAG index.)")
                 logger.info("RAG 索引处理完成。(RAG index handling complete.)")


            # 4. Graph Building
            with st.spinner("步骤 4/5: 正在构建依赖图... (Building dependency graph...)"):
                 if st.session_state.preprocessed_chunks:
                     st.session_state.graph = graph_builder.build_dependency_graph_v2(
                         st.session_state.preprocessed_chunks,
                         st.session_state.all_nodes_info or {},
                         st.session_state.all_imports_info or {}
                         # Pass RAG handler if implementing bootstrapping:
                         # rag_handler_instance=st.session_state.rag_handler_instance
                     )
                 if not st.session_state.graph or len(st.session_state.graph.nodes) == 0:
                     st.warning("未能构建有效的依赖图或图为空。(Failed to build graph or graph is empty.)")
                     st.session_state.graph = nx.DiGraph() # Ensure it's an empty graph object
                 logger.info("图构建完成。(Graph building complete.)")


            # Convert graph for agraph display
            st.session_state.agraph_nodes, st.session_state.agraph_edges = convert_nx_to_agraph(st.session_state.graph)


            # 5. Initial Project Report Generation
            with st.spinner("步骤 5/5: 正在生成项目报告... (Generating project report...)"):
                 if st.session_state.rag_handler_instance:
                      try:
                           report_query = "为这个 Python 项目生成一个高级别的技术概览报告，涵盖其主要模块、核心功能和潜在的架构模式。(Generate a high-level technical overview report for this Python project, covering main modules, core functionalities, and potential architectural patterns.)"
                           # Retrieve context relevant to the whole project (e.g., module summaries)
                           # Use RAG retrieve for broader context search
                           report_context_chunks = st.session_state.rag_handler_instance.retrieve("project overview structure modules functionality architecture", k=5)
                           if not report_context_chunks and st.session_state.preprocessed_chunks:
                                # Fallback: use module chunks directly if retrieval fails
                                report_context_chunks = [c for c in st.session_state.preprocessed_chunks if c.get('type') == 'module']

                           if report_context_chunks:
                                st.session_state.project_report = st.session_state.rag_handler_instance.generate_response(
                                    report_query,
                                    report_context_chunks
                                )
                           else:
                                # Try generating without specific context if none found
                                st.session_state.project_report = st.session_state.rag_handler_instance.generate_response(report_query, [])
                                st.info("未能找到生成报告的特定上下文，尝试直接生成。(No specific context found for report, attempting direct generation.)")

                           # Check if report generation itself indicated an error
                           if st.session_state.project_report.startswith("[生成回答时出错"):
                                st.session_state.project_report_error = st.session_state.project_report
                                st.session_state.project_report = "" # Clear report content
                      except Exception as report_err:
                           logger.error(f"生成项目报告失败: {report_err}", exc_info=True)
                           st.session_state.project_report_error = f"[生成项目报告时发生意外错误: {report_err}]"
                           st.session_state.project_report = ""


            # --- Analysis Complete ---
            st.session_state.analysis_done = True
            st.sidebar.success("分析完成！(Analysis Complete!)")
            st.rerun() # Rerun to update the UI state cleanly

        except Exception as e:
            # Catch any unexpected error during the pipeline
            logger.error(f"分析流程中发生错误 (Error during analysis pipeline): {e}", exc_info=True)
            st.error(f"分析失败 (Analysis failed): {e}")
            st.session_state.analysis_error = str(e)
            st.session_state.analysis_done = False
            st.rerun() # Rerun to show error state

    else:
        st.sidebar.warning("请输入有效的目录路径。(Please enter a valid directory path.)")


# --- Main Area Layout ---
col1, col2 = st.columns([3, 2]) # Graph left, Details/RAG right

# --- Left Column: Graph Display ---
with col1:
    st.subheader("项目结构图 (Project Structure Graph)")
    if st.session_state.analysis_done and st.session_state.agraph_nodes:
        # Configure Agraph (adjust physics/layout as needed)
        config = Config(width=1200, height=1000, directed=True, physics=True, hierarchical=False,
                        # Fine-tune physics for better layout stability
                        physics_options={
                            "enabled": True,
                            "barnesHut": {"gravitationalConstant": -8000, "springConstant": 0.04, "springLength": 150},
                            "minVelocity": 0.75,
                            # "solver": "forceAtlas2Based" # Experiment with solvers
                        },
                        interaction={"hover": True, "tooltipDelay": 300, "navigationButtons": True, "keyboard": True},
                        node={'labelProperty': 'label'},
                        edge={'labelProperty': 'label'}
                       )

        # Display the graph and capture clicked node ID
        clicked_node_id = agraph(nodes=st.session_state.agraph_nodes,
                                 edges=st.session_state.agraph_edges,
                                 config=config)

        # Update selected node in session state if a node was clicked
        if clicked_node_id and clicked_node_id != st.session_state.selected_node_id:
            st.session_state.selected_node_id = clicked_node_id
            # Clear previous RAG explanation when node changes
            st.session_state.last_rag_question = ""
            st.session_state.last_rag_answer = ""
            st.rerun() # Rerun to update the right panel immediately

    elif st.session_state.analysis_done:
        st.info("分析完成，但图中没有节点可显示。(Analysis complete, but no nodes to display in the graph.)")
    elif st.session_state.analysis_error:
         st.warning(f"分析失败，无法显示图表。错误：{st.session_state.analysis_error} (Analysis failed, cannot display graph.)")
    else:
        st.info("请在左侧选择项目并点击 '分析项目'。(Select a project and click 'Analyze Project' on the left.)")


# --- Right Column: Tabs for Details and RAG ---
with col2:
    if st.session_state.analysis_done:
        # Create Tabs
        tab1, tab2 = st.tabs(["📊 项目报告 (Project Report)", "💬 互动分析 (Interactive Analysis)"])

        # --- Tab 1: Project Report ---
        with tab1:
            st.subheader("自动项目报告 (Automated Project Report)")
            if st.session_state.project_report_error:
                 st.error(f"生成报告时出错 (Error generating report): {st.session_state.project_report_error}")
            elif st.session_state.project_report:
                st.markdown(st.session_state.project_report)
            else:
                st.info("正在生成或无可用报告。(Generating report or no report available.)")

            # Add a button to regenerate report manually
            if st.button("重新生成报告 (Regenerate Report)", key="regen_report"):
                 with st.spinner("正在重新生成项目报告... (Regenerating project report...)"):
                      st.session_state.project_report = "" # Clear previous report/error
                      st.session_state.project_report_error = None
                      if st.session_state.rag_handler_instance:
                           try:
                                report_query = "为这个 Python 项目生成一个高级别的技术概览报告，涵盖其主要模块、核心功能和潜在的架构模式。(Generate a high-level technical overview report for this Python project, covering main modules, core functionalities, and potential architectural patterns.)"
                                report_context_chunks = st.session_state.rag_handler_instance.retrieve("project overview structure modules functionality architecture", k=5)
                                if not report_context_chunks and st.session_state.preprocessed_chunks:
                                     report_context_chunks = [c for c in st.session_state.preprocessed_chunks if c.get('type') == 'module']

                                if report_context_chunks:
                                     st.session_state.project_report = st.session_state.rag_handler_instance.generate_response(report_query, report_context_chunks)
                                else:
                                     st.session_state.project_report = st.session_state.rag_handler_instance.generate_response(report_query, [])

                                if st.session_state.project_report.startswith("[生成回答时出错"):
                                     st.session_state.project_report_error = st.session_state.project_report
                                     st.session_state.project_report = ""
                           except Exception as report_err:
                                logger.error(f"重新生成项目报告失败: {report_err}", exc_info=True)
                                st.session_state.project_report_error = f"[重新生成报告时发生意外错误: {report_err}]"
                      else:
                           st.session_state.project_report_error = "RAG 系统未初始化。(RAG system not initialized.)"
                      st.rerun() # Update UI


        # --- Tab 2: Interactive Analysis ---
        with tab2:
            st.subheader("互动分析 (Interactive Analysis)")

            # Display selected node details
            selected_node_data = None
            if st.session_state.selected_node_id and st.session_state.graph:
                if st.session_state.selected_node_id in st.session_state.graph.nodes:
                    selected_node_data = st.session_state.graph.nodes[st.session_state.selected_node_id]
                    st.markdown(f"#### 选中节点 (Selected Node): `{selected_node_data.get('short_name', st.session_state.selected_node_id)}`")
                    st.markdown(f"**类型 (Type):** {selected_node_data.get('type', 'N/A')} | **文件 (File):** `{selected_node_data.get('filepath', 'N/A')}` | **行 (Line):** {selected_node_data.get('lineno', 'N/A')}")

                    # Context Management Button
                    is_in_context = st.session_state.selected_node_id in st.session_state.selected_context_ids
                    button_text = "从上下文中移除 (Remove from Context)" if is_in_context else "加入到上下文 (Add to Context)"
                    if st.button(button_text, key=f"ctx_btn_{st.session_state.selected_node_id}"):
                        if is_in_context:
                            st.session_state.selected_context_ids.remove(st.session_state.selected_node_id)
                        else:
                            # Avoid adding duplicates
                            if st.session_state.selected_node_id not in st.session_state.selected_context_ids:
                                 st.session_state.selected_context_ids.append(st.session_state.selected_node_id)
                        st.rerun() # Update UI

                    # Display details in expanders
                    with st.expander("摘要 (Summary)"):
                         summary = selected_node_data.get('summary')
                         st.markdown(summary if summary and not summary.startswith("[") else "*无有效摘要 (No valid summary)*")
                    with st.expander("代码片段 (Code Snippet)"):
                         st.code(selected_node_data.get('code', '# N/A'), language='python')
                    with st.expander("提取的关系 (Extracted Relationships)"):
                         rels = selected_node_data.get('relationships')
                         if rels:
                              st.json(rels)
                         else:
                              st.markdown("*未提取到关系或无关系。(No relationships extracted or none found.)*")
                    st.divider()
                else:
                    st.warning(f"选中的节点 ID '{st.session_state.selected_node_id}' 在图中未找到。请重新选择。(Selected node ID not found in graph. Please reselect.)")
                    st.session_state.selected_node_id = None # Reset selection

            else:
                 st.info("请在左侧图中点击一个节点以查看其详细信息。(Click a node in the graph on the left to see details.)")


            # Display Current Context
            st.markdown("---")
            st.markdown("**当前上下文 (Current Context):**")
            if st.session_state.selected_context_ids:
                 context_display_list = []
                 for ctx_id in st.session_state.selected_context_ids:
                      # Try to get node data for display name
                      ctx_node_data = None
                      if st.session_state.graph and ctx_id in st.session_state.graph.nodes:
                           ctx_node_data = st.session_state.graph.nodes[ctx_id]
                           display_name = f"`{ctx_node_data.get('short_name', ctx_id)}` ({ctx_node_data.get('type', 'N/A')})"
                      else:
                           # Fallback if graph data isn't available for the ID
                           display_name = f"`{ctx_id}`"
                      context_display_list.append(f"- {display_name}")
                 st.markdown("\n".join(context_display_list))

                 if st.button("清空上下文 (Clear Context)", key="clear_ctx"):
                      st.session_state.selected_context_ids = []
                      st.rerun()
            else:
                 st.markdown("*未选择任何上下文。将自动检索相关信息。(No context selected. Relevant information will be retrieved automatically.)*")


            # RAG Q&A Section
            st.markdown("---")
            st.markdown("**提问 (Ask Questions):**")
            question = st.text_area("输入您关于代码的问题 (Enter your question about the code):", key="rag_question_input", value=st.session_state.last_rag_question)
            ask_rag_button = st.button("提问 (Ask)", key="ask_rag")

            if ask_rag_button and question:
                 st.session_state.last_rag_question = question
                 st.session_state.last_rag_answer = "" # Clear previous answer
                 retrieved_chunks_for_q: List[Dict[str, Any]] = []

                 if not st.session_state.rag_handler_instance:
                      st.error("RAG 系统未初始化，无法回答问题。(RAG system not initialized.)")
                 else:
                      with st.spinner("正在检索相关代码块... (Retrieving relevant code chunks...)"):
                           try:
                               # If context is selected, retrieve those chunks directly
                               if st.session_state.selected_context_ids:
                                    logger.info(f"Using {len(st.session_state.selected_context_ids)} selected context nodes for RAG.")
                                    for ctx_id in st.session_state.selected_context_ids:
                                         # Use RAG handler to get full chunk data by ID
                                         chunk_data = st.session_state.rag_handler_instance.get_chunk_by_id(ctx_id)
                                         if chunk_data:
                                              retrieved_chunks_for_q.append(chunk_data)
                                         else:
                                              logger.warning(f"Could not retrieve chunk data for context ID: {ctx_id}")
                                    # Optionally, still perform a vector search to augment the context?
                                    # retrieved_chunks_for_q.extend(st.session_state.rag_handler_instance.retrieve(question, k=2))
                                    # Deduplicate if needed: retrieved_chunks_for_q = list({c['id']: c for c in retrieved_chunks_for_q}.values())
                               else:
                                    logger.info("No context selected, performing vector search for RAG.")
                                    retrieved_chunks_for_q = st.session_state.rag_handler_instance.retrieve(question, k=5) # Retrieve top 5

                           except Exception as retrieve_err:
                                st.error(f"检索时出错 (Error during retrieval): {retrieve_err}")
                                logger.error("Error during RAG retrieval", exc_info=True)

                      # Proceed to generation if retrieval was attempted (even if it returned empty)
                      with st.spinner("正在生成回答... (Generating answer...)"):
                           try:
                                st.session_state.last_rag_answer = st.session_state.rag_handler_instance.generate_response(
                                    question,
                                    retrieved_chunks_for_q # Pass retrieved chunks (could be empty)
                                )
                           except Exception as gen_err:
                                st.error(f"生成回答时出错 (Error during generation): {gen_err}")
                                logger.error("Error during RAG generation", exc_info=True)
                                st.session_state.last_rag_answer = "[生成回答时发生错误 (Error during generation)]"
                 st.rerun() # Update UI to show answer or errors

            # Display last Q&A
            if st.session_state.last_rag_answer:
                 st.markdown("**回答 (Answer):**")
                 st.markdown(st.session_state.last_rag_answer)

    elif st.session_state.analysis_error:
         # Show error prominently if analysis failed before completion
         st.error(f"分析未能完成，请查看侧边栏错误信息。(Analysis could not complete. See error in sidebar.)")
    else:
        # Initial state before analysis
        st.info("请在左侧选择项目并点击 '分析项目' 以启动。(Select a project and click 'Analyze Project' on the left to start.)")

