# 文件名: demo.py
import sys
import os
import gradio as gr
import logging
from typing import List, Tuple

# --- 统一且健壮的路径设置 ---
_current_script_path = os.path.abspath(__file__)
_chat_dir = os.path.dirname(_current_script_path)
PROJECT_ROOT_DIR = os.path.dirname(_chat_dir)

# 将项目根目录添加到Python解释器的模块搜索路径中
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- 导入自定义模块 ---
try:
    from chat.main import State, app, agent, embed_model, similarity
    from Interaction_Design.user_memory import Retriever_Memory
    from Interaction_Design.AI_emotion_status import Score
    from Interaction_Design.chat_history import chat_history
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    sys.exit(1)

# --- 日志配置 ---
LOGS_DIR = os.path.join(PROJECT_ROOT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'demo_log.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 初始化 AI 评分模块
AI_score = Score()

# --- 全局变量存储上下文 ---
# 使用字典存储每个会话的上下文
session_contexts = {}


def get_or_create_context(session_id: str) -> dict:
    """获取或创建会话上下文"""
    if session_id not in session_contexts:
        session_contexts[session_id] = {
            'history': [],  # 聊天历史
            'last_user_input': '',  # 上一次用户输入
            'initialized': True
        }
    return session_contexts[session_id]


def chat_function(message: str, history: List[Tuple[str, str]], session_id: str) -> List[Tuple[str, str]]:
    """
    处理用户消息并返回回复
    
    Args:
        message: 用户当前输入
        history: Gradio 聊天历史 [(user_msg, bot_msg), ...]
        session_id: 会话ID
    
    Returns:
        更新后的历史
    """
    if not session_id or not session_id.strip():
        return history + [(message, "⚠️ 请先输入会话ID！")]
    
    if not message or not message.strip():
        return history
    
    try:
        logger.info(f"[Session: {session_id}] User Input: {message}")
        
        # 获取会话上下文
        context = get_or_create_context(session_id)
        
        # 判断相似度（如果不是第一条消息）
        if context['last_user_input']:
            user_process = similarity.process(message, session_id, 3)
            
            if user_process and len(user_process) > 0:
                score = user_process[0][-1]
                logger.info(f"[Session: {session_id}] Similarity score: {score}")
                
                if score <= 0.85:
                    # 相似度低，使用常规检索
                    long_term = agent.search_memory(message, embed_model, session_id, 3)
                else:
                    # 相似度高，使用特定检索
                    retriever_Memory = Retriever_Memory(embed_model, session_id)
                    long_term = retriever_Memory.retriever(message, 3)
            else:
                # 如果没有相似度结果，使用常规检索
                long_term = agent.search_memory(message, embed_model, session_id, 3)
        else:
            # 第一次对话，直接检索
            long_term = agent.search_memory(message, embed_model, session_id, 2)
        
        # 构建状态
        message_state = State(
            input=message,
            long_term=long_term,
            session_id=session_id,
            answer=''
        )
        
        # 调用 LangGraph
        logger.info(f"[Session: {session_id}] Invoking LangGraph...")
        result = app.invoke(message_state)
        assistant_response = result.get('answer', '嗯...我好像不知道该说什么了。')
        
        logger.info(f"[Session: {session_id}] Model Response: {assistant_response}")
        
        # 保存聊天历史
        if context['last_user_input']:
            chat_history(context['last_user_input'], assistant_response, session_id)
        
        # 更新上下文
        context['last_user_input'] = message
        
        # 更新 AI 情绪评分
        try:
            AI_score.main(session_id)
        except Exception as e:
            logger.warning(f"[Session: {session_id}] AI score update failed: {e}")
        
        # 返回更新后的历史
        return history + [(message, assistant_response)]
        
    except Exception as e:
        error_message = f"处理消息时发生错误: {str(e)}"
        logger.error(f"[Session: {session_id}] Error: {error_message}", exc_info=True)
        return history + [(message, f"抱歉，处理时出错了：{e}")]


def clear_session(session_id: str) -> List[Tuple[str, str]]:
    """清除会话上下文并返回空历史"""
    if session_id in session_contexts:
        del session_contexts[session_id]
        logger.info(f"[Session: {session_id}] Context cleared")
    return []  # 返回空的聊天历史


# --- Gradio 界面 ---
with gr.Blocks(title="你的专属伴侣❤️", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 你的专属伴侣 ❤️")
    gr.Markdown("和你的虚拟伴侣开始聊天吧！")
    
    with gr.Row():
        with gr.Column(scale=3):
            session_id_input = gr.Textbox(
                label="会话ID",
                placeholder="请输入你的专属会话ID (例如 'user123')",
                value="",
                interactive=True
            )
        with gr.Column(scale=1):
            clear_btn = gr.Button("清除会话", variant="secondary")
    
    chatbot = gr.Chatbot(
        label="聊天窗口",
        height=500,
        show_copy_button=True
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="输入消息",
            placeholder="你想对我说什么？",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)
    
    # 绑定事件
    submit_btn.click(
        fn=chat_function,
        inputs=[msg, chatbot, session_id_input],
        outputs=chatbot
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=None,
        outputs=msg
    )
    
    msg.submit(
        fn=chat_function,
        inputs=[msg, chatbot, session_id_input],
        outputs=chatbot
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=None,
        outputs=msg
    )
    
    clear_btn.click(
        fn=clear_session,
        inputs=session_id_input,
        outputs=chatbot
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=None,
        outputs=msg
    )
    
    gr.Markdown("""
    ### 使用说明：
    1. 输入你的专属会话ID（例如：user123）
    2. 在下方输入框输入消息
    3. 点击"发送"或按Enter键发送消息
    4. 如需重新开始，点击"清除会话"按钮
    """)


if __name__ == "__main__":
    logger.info("Starting Gradio demo...")
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=True,  # 设置为True可以生成公开链接
        show_error=True
    )
