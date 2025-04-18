# app.py
# -*- coding: utf-8 -*-

"""
(V2 Refactor + Filetree/README) Streamlit 主应用程序，整合了重构后的模块。
- 使用 src/ 目录下的代码解析、预处理、图构建和 RAG 模块。
- RAG Handler 负责处理项目特定的索引/缓存和 LLM 交互。
- 支持通过配置选择不同的 LLM 提供商和模型。
- 新增：显示文件树结构和 README 内容。
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

# --- Helper Function for Agraph Conversion (Updated Styles) ---
def convert_nx_to_agraph(graph: nx.DiGraph) -> Tuple[List[Node], List[Edge]]:
    """将 NetworkX 图转换为 streamlit-agraph 需要的节点和边列表。"""
    agraph_nodes = []
    agraph_edges = []

    if not graph:
        return agraph_nodes, agraph_edges

    # --- Define styles for new node types ---
    node_styles = {
        'project': {'color': '#FFD700', 'shape': 'star', 'size': 30},
        'directory': {'color': '#C0C0C0', 'shape': 'box', 'size': 20},
        'file': {'color': '#ADD8E6', 'shape': 'ellipse', 'size': 10},
        'readme': {'color': '#98FB98', 'shape': 'box', 'size': 15},
        'function': {'color': '#74A9D8', 'shape': 'dot', 'size': 15},
        'class': {'color': '#F0A367', 'shape': 'box', 'size': 20},
        'module': {'color': '#90EE90', 'shape': 'database', 'size': 25},
        'unknown': {'color': '#CCCCCC', 'shape': 'ellipse', 'size': 10}
    }

    # --- Edge styles ---
    edge_styles = {
        'CONTAINS': {'color': '#E0E0E0', 'dashes': True, 'label': ''},
        'DEFINES': {'color': '#B0E0E6', 'dashes': False, 'label': ''},
        'CALLS': {'color': '#ADD8E6', 'dashes': False},
        'DEPENDS_ON': {'color': '#FFA07A', 'dashes': True},
        'IMPLEMENTS': {'color': '#FFB6C1', 'dashes': False},
        'RELATES_TO': {'color': '#D3D3D3', 'dashes': True},
        'MODULE_SUMMARY_FOR': {'color': '#90EE90', 'dashes': True, 'label': 'summary'},
    }
    default_edge_color = "#CCCCCC"


    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        style = node_styles.get(node_type, node_styles['unknown'])
        node_label = str(node_data.get('short_name', node_data.get('label', node_id)))

        agraph_nodes.append(Node(id=node_id,
                                 label=node_label,
                                 title=node_data.get('title', ''),
                                 size=style['size'],
                                 shape=style['shape'],
                                 color=style['color']
                                 ))

    for u, v, edge_data in graph.edges(data=True):
        if u not in graph or v not in graph:
            logger.warning(f"Skipping edge ({u} -> {v}) because one or both nodes do not exist in the graph node set.")
            continue

        edge_type = edge_data.get('type', 'UNKNOWN')
        style = edge_styles.get(edge_type, {'color': default_edge_color, 'dashes': False, 'label': edge_type})
        edge_label = str(style.get('label', edge_data.get('label', '')))

        agraph_edges.append(Edge(source=u,
                                 target=v,
                                 label=edge_label,
                                 type="CURVE_SMOOTH",
                                 color=style['color'],
                                 dashes=style['dashes']
                                 ))

    return agraph_nodes, agraph_edges


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Python 代码分析器 (Filetree Enhanced)")
st.title("🐍 Python AST & RAG 代码分析器 (Filetree Enhanced)")

# --- Session State Initialization ---
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.project_dir = "."
    st.session_state.analysis_done = False
    st.session_state.analysis_error = None
    st.session_state.all_nodes_info = None
    st.session_state.all_imports_info = None
    st.session_state.file_tree = None
    st.session_state.readme_content = None
    st.session_state.preprocessed_chunks = None
    st.session_state.graph = None
    st.session_state.agraph_nodes = []
    st.session_state.agraph_edges = []
    st.session_state.rag_handler_instance = None
    st.session_state.selected_node_id = None
    st.session_state.selected_context_ids = []
    st.session_state.project_report = ""
    st.session_state.project_report_error = None
    st.session_state.last_rag_question = ""
    st.session_state.last_rag_answer = ""
    logger.info("Streamlit session state initialized.")


# --- Sidebar ---
st.sidebar.header("项目选择 (Project Selection)")
project_dir_input = st.sidebar.text_input("输入 Python 项目目录路径:", st.session_state.project_dir)
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
        st.session_state.project_dir = abs_project_dir
        # Reset state for new analysis
        st.session_state.analysis_done = False
        st.session_state.analysis_error = None
        st.session_state.all_nodes_info = None
        st.session_state.all_imports_info = None
        st.session_state.file_tree = None
        st.session_state.readme_content = None
        st.session_state.preprocessed_chunks = None
        st.session_state.graph = None
        st.session_state.agraph_nodes = []
        st.session_state.agraph_edges = []
        st.session_state.rag_handler_instance = None
        st.session_state.selected_node_id = None
        st.session_state.selected_context_ids = []
        st.session_state.project_report = ""
        st.session_state.project_report_error = None
        st.session_state.last_rag_question = ""
        st.session_state.last_rag_answer = ""

        st.sidebar.info(f"开始分析: {st.session_state.project_dir}")
        logger.info(f"Starting analysis for project: {st.session_state.project_dir}")

        # --- Execute Full Pipeline ---
        try:
            # 0. Initialize RAG Handler
            with st.spinner("步骤 0/5: 初始化 RAG 系统..."):
                 try:
                      st.session_state.rag_handler_instance = RAGHandler(project_path=st.session_state.project_dir)
                      project_data_dir = st.session_state.rag_handler_instance.project_data_dir
                      preprocessed_data_path = os.path.join(project_data_dir, PREPROCESSED_DATA_FILENAME)
                      logger.info("RAG Handler initialized.")
                 except ValueError as rag_init_err:
                      st.error(f"无法初始化 RAG 系统: {rag_init_err}")
                      st.session_state.analysis_error = f"RAG Init failed: {rag_init_err}"
                      st.stop()


            # 1. Parsing (Updated Call)
            with st.spinner("步骤 1/5: 正在解析 Python 文件和扫描文件树..."):
                nodes, imports, tree, readme = code_parser.parse_project(st.session_state.project_dir)
                st.session_state.all_nodes_info = nodes
                st.session_state.all_imports_info = imports
                st.session_state.file_tree = tree
                st.session_state.readme_content = readme

                if not st.session_state.all_nodes_info and not st.session_state.all_imports_info:
                    st.warning("解析项目时未找到任何有效的 Python 节点或导入。")
                logger.info(f"解析和文件树扫描完成。")


            # 2. Preprocessing
            with st.spinner("步骤 2/5: 正在预处理代码块 (可能需要几分钟)..."):
                 loaded_from_cache = False
                 if os.path.exists(preprocessed_data_path):
                      logger.info(f"尝试从缓存加载预处理数据: {preprocessed_data_path}")
                      try:
                          with open(preprocessed_data_path, 'rb') as f:
                              st.session_state.preprocessed_chunks = pickle.load(f)
                          logger.info(f"成功加载 {len(st.session_state.preprocessed_chunks)} 个预处理块。")
                          loaded_from_cache = True
                      except Exception as load_err:
                           logger.warning(f"加载预处理缓存失败: {load_err}。将重新生成。")
                           st.session_state.preprocessed_chunks = None

                 if not loaded_from_cache:
                      logger.info("执行 preprocessing.preprocess_project...")
                      st.session_state.preprocessed_chunks = preprocessing.preprocess_project(
                          st.session_state.all_nodes_info or {},
                          st.session_state.all_imports_info or {}
                      )
                      if not st.session_state.preprocessed_chunks:
                           st.warning("预处理步骤未生成任何代码块。")
                      else:
                           try:
                                with open(preprocessed_data_path, 'wb') as f:
                                     pickle.dump(st.session_state.preprocessed_chunks, f)
                                logger.info(f"预处理结果已保存到缓存: {preprocessed_data_path}")
                           except Exception as save_err:
                                logger.error(f"保存预处理结果失败: {save_err}")
                 logger.info("预处理完成。")


            # 3. RAG Indexing
            with st.spinner("步骤 3/5: 正在构建/加载 RAG 索引..."):
                 if st.session_state.rag_handler_instance and st.session_state.preprocessed_chunks:
                      st.session_state.rag_handler_instance.build_index(
                          st.session_state.preprocessed_chunks,
                          force_rebuild=True
                      )
                      if st.session_state.rag_handler_instance.index is None:
                           st.warning("构建 RAG 索引失败或索引为空。")
                 elif not st.session_state.preprocessed_chunks:
                      st.info("没有预处理数据可用于构建 RAG 索引。")
                 logger.info("RAG 索引处理完成。")


            # 4. Graph Building (Updated Call)
            with st.spinner("步骤 4/5: 正在构建依赖图 (含文件结构)..."):
                 if st.session_state.preprocessed_chunks is not None:
                     st.session_state.graph = graph_builder.build_dependency_graph_v2(
                         st.session_state.preprocessed_chunks,
                         st.session_state.all_nodes_info or {},
                         st.session_state.all_imports_info or {},
                         st.session_state.file_tree or [],
                         st.session_state.readme_content,
                         st.session_state.project_dir
                     )
                 else:
                      st.warning("预处理步骤未生成有效数据块，无法构建详细图谱。仅显示文件结构。")
                      st.session_state.graph = graph_builder.build_dependency_graph_v2(
                          [], {}, {},
                          st.session_state.file_tree or [],
                          st.session_state.readme_content,
                          st.session_state.project_dir
                      )

                 if not st.session_state.graph or len(st.session_state.graph.nodes) == 0:
                     st.warning("未能构建有效的依赖图或图为空。")
                     st.session_state.graph = nx.DiGraph()
                 logger.info("图构建完成。")


            # Convert graph for agraph display
            st.session_state.agraph_nodes, st.session_state.agraph_edges = convert_nx_to_agraph(st.session_state.graph)


            # 5. Initial Project Report Generation (Enhanced with README context)
            with st.spinner("步骤 5/5: 正在生成项目报告..."):
                 if st.session_state.rag_handler_instance:
                    try:
                        report_query = "为这个 Python 项目生成一个高级别的技术概览报告，涵盖其主要模块、核心功能和潜在的架构模式。(Generate a high-level technical overview report for this Python project, covering main modules, core functionalities, and potential architectural patterns.)"
                        base_context_query = "project overview structure modules functionality architecture"
                        report_context_chunks = st.session_state.rag_handler_instance.retrieve(base_context_query, k=4)

                        # --- Enhancement: Add README to context ---
                        if st.session_state.readme_content:
                            readme_node_id = next((nid for nid, data in st.session_state.graph.nodes(data=True) if data.get('type') == 'readme'), 'README.md')
                            readme_chunk_for_rag = {
                                'id': readme_node_id,
                                'filepath': 'README.md',
                                'type': 'readme',
                                'summary': '项目自述文件 (Project Readme file)',
                                'code': st.session_state.readme_content[:1000] + "..."
                            }
                            # Prepend README context for higher priority
                            report_context_chunks.insert(0, readme_chunk_for_rag) # *** Corrected Indentation Here ***
                            logger.info("Added README content to project report context.")
                        # --- End Enhancement ---

                        # --- Corrected Indentation for the following block ---
                        if report_context_chunks:
                            st.session_state.project_report = st.session_state.rag_handler_instance.generate_response(
                                report_query,
                                report_context_chunks
                            )
                        else:
                            st.session_state.project_report = st.session_state.rag_handler_instance.generate_response(report_query, [])
                            st.info("未能找到生成报告的特定上下文，尝试直接生成。")

                        if st.session_state.project_report.startswith("[生成回答时出错"):
                            st.session_state.project_report_error = st.session_state.project_report
                            st.session_state.project_report = ""

                    except Exception as report_err:
                        logger.error(f"生成项目报告失败: {report_err}", exc_info=True)
                        st.session_state.project_report_error = f"[生成项目报告时发生意外错误: {report_err}]"
                        st.session_state.project_report = ""
                 # --- End Corrected Indentation ---


            # --- Analysis Complete ---
            st.session_state.analysis_done = True
            st.sidebar.success("分析完成！(Analysis Complete!)")
            st.rerun()

        except Exception as e:
            logger.error(f"分析流程中发生错误: {e}", exc_info=True)
            st.error(f"分析失败: {e}")
            st.session_state.analysis_error = str(e)
            st.session_state.analysis_done = False
            st.rerun()

    else:
        st.sidebar.warning("请输入有效的目录路径。(Please enter a valid directory path.)")


# --- Main Area Layout ---
col1, col2 = st.columns([3, 2])

# --- Left Column: Graph Display ---
with col1:
    st.subheader("项目结构图 (含文件树)")
    if st.session_state.analysis_done and st.session_state.agraph_nodes:
        config = Config(width=1200, height=1000, directed=True, physics=True, hierarchical=False,
                        physics_options={
                            "enabled": True,
                            "barnesHut": {"gravitationalConstant": -10000, "springConstant": 0.05, "springLength": 200},
                            "minVelocity": 0.75,
                        },
                        interaction={"hover": True, "tooltipDelay": 300, "navigationButtons": True, "keyboard": True},
                        node={'labelProperty': 'label'},
                        edge={'labelProperty': 'label'}
                       )

        clicked_node_id = agraph(nodes=st.session_state.agraph_nodes,
                                 edges=st.session_state.agraph_edges,
                                 config=config)

        if clicked_node_id and clicked_node_id != st.session_state.selected_node_id:
            st.session_state.selected_node_id = clicked_node_id
            st.session_state.last_rag_question = ""
            st.session_state.last_rag_answer = ""
            st.rerun()

    elif st.session_state.analysis_done:
        st.info("分析完成，但图中没有节点可显示。")
    elif st.session_state.analysis_error:
         st.warning(f"分析失败，无法显示图表。错误：{st.session_state.analysis_error}")
    else:
        st.info("请在左侧选择项目并点击 '分析项目'。")


# --- Right Column: Tabs for Details and RAG ---
with col2:
    if st.session_state.analysis_done:
        tab1, tab2 = st.tabs(["📊 项目报告 & README", "💬 互动分析"])

        # --- Tab 1: Project Report & README ---
        with tab1:
            st.subheader("自动项目报告")
            if st.session_state.project_report_error:
                 st.error(f"生成报告时出错: {st.session_state.project_report_error}")
            elif st.session_state.project_report:
                st.markdown(st.session_state.project_report)
            else:
                st.info("正在生成或无可用报告。")

            # --- Display README ---
            st.markdown("---")
            st.subheader("README.md 内容")
            readme_content_display = "*未找到 README.md 或内容不可用*"
            if st.session_state.graph:
                readme_node_id = next((nid for nid, data in st.session_state.graph.nodes(data=True) if data.get('type') == 'readme'), None)
                if readme_node_id and readme_node_id in st.session_state.graph.nodes:
                    readme_content_display = st.session_state.graph.nodes[readme_node_id].get('content', readme_content_display)

            if readme_content_display == "*未找到 README.md 或内容不可用*" and st.session_state.readme_content:
                readme_content_display = st.session_state.readme_content

            if readme_content_display != "*未找到 README.md 或内容不可用*":
                 st.markdown(readme_content_display, unsafe_allow_html=True)
            else:
                 st.info(readme_content_display)
            # --- End README Display ---


        # --- Tab 2: Interactive Analysis ---
        with tab2:
            st.subheader("互动分析")

            # Display selected node details
            selected_node_data = None
            if st.session_state.selected_node_id and st.session_state.graph:
                if st.session_state.selected_node_id in st.session_state.graph.nodes:
                    selected_node_data = st.session_state.graph.nodes[st.session_state.selected_node_id]
                    node_type_display = selected_node_data.get('type', 'N/A')
                    node_path_display = selected_node_data.get('filepath', st.session_state.selected_node_id)

                    st.markdown(f"#### 选中节点: `{selected_node_data.get('label', st.session_state.selected_node_id)}`")
                    st.markdown(f"**类型:** {node_type_display} | **路径:** `{node_path_display}`")
                    if 'lineno' in selected_node_data:
                        st.markdown(f"**行:** {selected_node_data.get('lineno', 'N/A')}")

                    # --- Context Management Button ---
                    can_add_to_context = node_type_display not in ['project', 'directory', 'file']
                    if can_add_to_context:
                         is_in_context = st.session_state.selected_node_id in st.session_state.selected_context_ids
                         button_text = "从上下文中移除" if is_in_context else "加入到上下文"
                         if st.button(button_text, key=f"ctx_btn_{st.session_state.selected_node_id}"):
                             if is_in_context:
                                 st.session_state.selected_context_ids.remove(st.session_state.selected_node_id)
                             else:
                                 if st.session_state.selected_node_id not in st.session_state.selected_context_ids:
                                      st.session_state.selected_context_ids.append(st.session_state.selected_node_id)
                             st.rerun()
                    # --- End Context Management ---

                    # --- Display details in expanders ---
                    if node_type_display in ['function', 'class', 'module']:
                         with st.expander("摘要 (Summary)"):
                              summary = selected_node_data.get('summary')
                              st.markdown(summary if summary and not summary.startswith("[") else "*无有效摘要*")
                         with st.expander("代码片段 (Code Snippet)"):
                              st.code(selected_node_data.get('code', '# N/A'), language='python')
                         with st.expander("提取的关系 (Extracted Relationships)"):
                              rels = selected_node_data.get('relationships')
                              if rels: st.json(rels)
                              else: st.markdown("*未提取到关系或无关系。*")
                    elif node_type_display == 'readme':
                         with st.expander("README 内容预览"):
                              st.markdown(selected_node_data.get('content', '*内容不可用*')[:1000] + "...")

                    st.divider()
                else:
                    st.warning(f"选中的节点 ID '{st.session_state.selected_node_id}' 在图中未找到。请重新选择。")
                    st.session_state.selected_node_id = None

            else:
                 st.info("请在左侧图中点击一个节点以查看其详细信息。")


            # Display Current Context
            st.markdown("---")
            st.markdown("**当前上下文:**")
            if st.session_state.selected_context_ids:
                 context_display_list = []
                 for ctx_id in st.session_state.selected_context_ids:
                      ctx_node_data = None
                      if st.session_state.graph and ctx_id in st.session_state.graph.nodes:
                           ctx_node_data = st.session_state.graph.nodes[ctx_id]
                           display_name = f"`{ctx_node_data.get('label', ctx_id)}` ({ctx_node_data.get('type', 'N/A')})"
                      else:
                           display_name = f"`{ctx_id}`"
                      context_display_list.append(f"- {display_name}")
                 st.markdown("\n".join(context_display_list))

                 if st.button("清空上下文", key="clear_ctx"):
                      st.session_state.selected_context_ids = []
                      st.rerun()
            else:
                 st.markdown("*未选择任何上下文。将自动检索相关信息。*")


            # RAG Q&A Section
            st.markdown("---")
            st.markdown("**提问:**")
            question = st.text_area("输入您关于代码的问题:", key="rag_question_input", value=st.session_state.last_rag_question)
            ask_rag_button = st.button("提问 (Ask)", key="ask_rag")

            if ask_rag_button and question:
                 st.session_state.last_rag_question = question
                 st.session_state.last_rag_answer = ""
                 retrieved_chunks_for_q: List[Dict[str, Any]] = []

                 if not st.session_state.rag_handler_instance:
                      st.error("RAG 系统未初始化，无法回答问题。")
                 else:
                      with st.spinner("正在检索相关代码块..."):
                           try:
                               if st.session_state.selected_context_ids:
                                    logger.info(f"Using {len(st.session_state.selected_context_ids)} selected context nodes for RAG.")
                                    for ctx_id in st.session_state.selected_context_ids:
                                         chunk_data = st.session_state.rag_handler_instance.get_chunk_by_id(ctx_id)
                                         if chunk_data:
                                              retrieved_chunks_for_q.append(chunk_data)
                                              logger.debug(f"Added selected context chunk: {ctx_id}")
                                         else:
                                              if ctx_id in st.session_state.graph.nodes:
                                                   node_data = st.session_state.graph.nodes[ctx_id]
                                                   if node_data.get('type') == 'readme':
                                                         retrieved_chunks_for_q.append({
                                                             'id': ctx_id, 'filepath': ctx_id, 'type': 'readme',
                                                             'summary':'项目自述文件', 'code': node_data.get('content', '')[:1000]
                                                         })
                                                         logger.debug(f"Added selected context node (README): {ctx_id}")
                                              else:
                                                   logger.warning(f"Could not retrieve chunk data or node data for context ID: {ctx_id}")
                               else:
                                    logger.info("No context selected, performing vector search for RAG.")
                                    retrieved_chunks_for_q = st.session_state.rag_handler_instance.retrieve(question, k=5)

                           except Exception as retrieve_err:
                                st.error(f"检索时出错: {retrieve_err}")
                                logger.error("Error during RAG retrieval", exc_info=True)

                      with st.spinner("正在生成回答..."):
                           try:
                                if not retrieved_chunks_for_q:
                                     logger.warning("No context chunks found or selected for generation.")

                                st.session_state.last_rag_answer = st.session_state.rag_handler_instance.generate_response(
                                    question,
                                    retrieved_chunks_for_q
                                )
                           except Exception as gen_err:
                                st.error(f"生成回答时出错: {gen_err}")
                                logger.error("Error during RAG generation", exc_info=True)
                                st.session_state.last_rag_answer = "[生成回答时发生错误]"
                 st.rerun()

            # Display last Q&A
            if st.session_state.last_rag_answer:
                 st.markdown("**回答:**")
                 st.markdown(st.session_state.last_rag_answer)

    elif st.session_state.analysis_error:
         st.error(f"分析未能完成，请查看侧边栏错误信息。")
    else:
        st.info("请在左侧选择项目并点击 '分析项目' 以启动。")