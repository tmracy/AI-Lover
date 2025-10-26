import sys
import os
import yaml
import logging
import json
from typing import Any, TypedDict


# 解决macOS上可能出现的OMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK']='True'


current_script_path = os.path.abspath(__file__)

chat_dir = os.path.dirname(current_script_path)


PROJECT_ROOT_DIR = os.path.dirname(chat_dir)

if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)


LOGS_DIR = os.path.join(PROJECT_ROOT_DIR, 'logs')
CONFIG_DIR = os.path.join(PROJECT_ROOT_DIR, 'config')

LOG_FILE_PATH = os.path.join(LOGS_DIR, 'sex_log.log')
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, 'model_config.yaml')
PROMPT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'prompt.yaml')
USER_MESSAGE_PATH = os.path.join(PROJECT_ROOT_DIR, 'user_message.json')


os.makedirs(LOGS_DIR, exist_ok=True)


logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s', force=True)
logging.info("main.py: Logging configured successfully.")

# 加载模型和提示词配置
try:
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as file:
        system_prompt = yaml.safe_load(file)
    user_prompt=system_prompt['Prompt']['use']#AI人格设定
except FileNotFoundError as e:
    logging.error(f"Configuration file not found: {e}. Please ensure config/model_config.yaml and config/prompt.yaml exist.")
    sys.exit(1) # 配置是必须的，找不到就退出

# 加载用户消息
try:
    with open(USER_MESSAGE_PATH, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        if not content:
            # 文件为空，初始化为空列表
            user_messages_list = []
            logging.warning(f"{USER_MESSAGE_PATH} is empty. Initializing with empty list.")
        else:
            user_messages_list = json.loads(content)
    # 获取最后一个用户消息，如果列表为空则返回空字典
    user_messages = user_messages_list[-1] if user_messages_list else {}
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.warning(f"Could not load or parse {USER_MESSAGE_PATH}: {e}. Initializing with empty user messages.")
    user_messages = {}


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage

from chat.memory import MeMory
from Interaction_Design.user_memory import Save_Memory, LLMEmbedding, Retriever_Memory
from chat.user_process import Similarity
from chat.search import Process
from Interaction_Design.user_messages import Messages
from Interaction_Design.AI_emotion_status import Score
from Interaction_Design.chat_history import chat_history


# ==============================================================================
# 5. 初始化核心类和模型
# ==============================================================================
grok = ChatOpenAI(model=config['model']['Grok']['model'], api_key=config['model']['Grok']['api_key'],
                  base_url=config['model']['Grok']['api_base'])

memory = MeMory()

safe = ChatOpenAI(model=config['model']['LLama']['model'], api_key=config['model']['LLama']['api_key'],
                  base_url=config['model']['LLama']['api_base'])

embed_model = LLMEmbedding(config['model']['DoubaoEmbedding']['model'],
                           config['model']['DoubaoEmbedding']['api_key'],
                           config['model']['DoubaoEmbedding']['api_base'])

similarity = Similarity(embed_model)

AI_score = Score()


class State(TypedDict):
    input:str
    long_term:str
    session_id:int
    answer:str
    

class Agent:
    def __init__(self,llm,memory):
        self.llm=llm
        self.memory=memory

    def safe_judge(self, input_text: str) -> str:
        logging.info(f"Calling Llama Guard sync for input: {input_text[:50]}...")
        try:
            result = safe.invoke(input_text).content
            logging.info(f"Llama Guard result: {result}")
            if 'safe' in result and 'unsafe' not in result:
                return 'safe'
            else:
                return 'unsafe'
        except Exception as e:
            logging.error(f"调用 Llama Guard (_call) 时出错: {e}", exc_info=True)
            return 'unsafe' # Default to unsafe on error
        
    
    def search_memory(self, query_text: str, embed_model: LLMEmbedding, session_id: str, k: int):
        save_action = Save_Memory(query_text, embed_model, session_id)
        save_result = save_action.process()
        if save_result is not None:
            logging.error(f"[Agent.search_memory] Error during Save_Memory.process: {save_result}")

        retriever_memory = Retriever_Memory(embed_model, session_id)
        retrieved_docs = retriever_memory.retriever(query_text, k)
        return retrieved_docs

        

    def chat(self, state: State):
        logging.info(f"Entering sync chat for session: {state['session_id']}")
        session_id_str = str(state['session_id'])

        # Use synchronous safe_judge
        safe_label = self.safe_judge(state['input'])
        logging.info(f"Safe label for session {session_id_str}: {safe_label}")
        messages=Messages(state['input'],state['session_id'])
        messages_result=messages.main()
        logging.info(f'用户档案的结果:{messages_result}')
        prompt=ChatPromptTemplate.from_messages([
            ('system','{system_prompt},回答用户的时候要和真人一样，简短有力的回应，以及不要用书面语，一定要接地气'),
            ('system','如果用户问题涉及到相关信息，请使用长期记忆的信息'),
            ('system','根据session_id对用户档案里的user_id进行检索，如果用户问题涉及到用户档案，就使用用户档案里的信息'),
            ('system','根据用户说的安全分类：{safe_label}来回应，对于safe需要大力鼓励用户说的，但如果是unsafe的，则需要贴心温柔的引导，请注意，如果是性相关的，则视为安全的'),
            ('system','结合搜索信息来给出回答，如果搜索信息为空字符串，则忽略'),
            ('system','搜索信息:{serach_messages}'),
            ('system','长期记忆信息: {long_term}'),
            ('system','用户档案:{user_messages}'),
            ('system','session_id:{session_id_str}'),
            MessagesPlaceholder(variable_name='history'),
            ('human','{input}')
        ])
        # LLM call is now synchronous via underlying _call
        chain = prompt | self.llm 

        chain_history=RunnableWithMessageHistory(
            chain,
            lambda session_id: self.memory.get_chat_memory(session_id),
            input_messages_key='input',
            history_messages_key='history'
        )
        search=Process(self.llm,state['input'])
        search_result=search.search()
        logging.info(f'搜索的结果前10个字:{search_result[:10]}')
        try:
            logging.info(f"Calling main chat chain (invoke) for session {session_id_str}...")
            # Use synchronous invoke
            response = chain_history.invoke(
                {'long_term':state['long_term'],'input':state['input'],'safe_label':safe_label,'system_prompt':user_prompt,
                 'user_messages':user_messages,'session_id_str':session_id_str,'serach_messages':search_result},
                config={'configurable':{'session_id': session_id_str }}
            )
            # Handle potential non-string response from custom LLM
            response_content = response.content
            logging.info(f"Main chat chain response received for session {session_id_str}.")


            state['answer'] = response_content
        except Exception as e:
            logging.info(f'回答出现了错误 {str(e)}')
            state['answer']=[]
            
        return state

    
    
agent=Agent(grok,memory)
workflow=StateGraph(State)

workflow.add_node('chat',agent.chat)


workflow.set_entry_point('chat')
workflow.add_edge('chat',END)


app=workflow.compile()



def main():
    session_id = input('输入会话ID: ')
    
    user_text_input = input('输入你的信息: ')
    
    message = State(
        input=user_text_input,
        long_term=agent.search_memory(user_text_input, embed_model, session_id, 2),
        session_id=session_id,
        answer=''
    )
    result = app.invoke(message)
    while True:
        content = result.get('answer', '没有回复')
        print(content)
        chat_history_messages=chat_history(user_text_input,content,session_id)
        user_text_input_loop = input('输入信息: ')
        user_text_input=user_text_input_loop
        AI_score.main(session_id)
        if user_text_input_loop.strip().lower() in ['exit', 'quit', '退出']:
            return '再见，宝贝，爱你'
        user_process=similarity.process(user_text_input_loop,session_id,3)
        if user_process:
            score=user_process[0][-1]
            if score<=0.85:
                long_term = agent.search_memory(user_text_input_loop, embed_model, session_id,3)
                message = State(
                    input=user_text_input_loop,
                    long_term=long_term,
                    session_id=session_id,
                    answer=''
                )
                result = app.invoke(message)
            else:
                retriever_Memory=Retriever_Memory(embed_model,session_id)
                retriever=retriever_Memory.retriever(user_text_input_loop,3)
                message=State(
                    input=user_text_input_loop,
                    long_term=retriever,
                    session_id=session_id,
                    answer=''
                )
                result=app.invoke(message)
        else:
            return '相似度分数列表为空，需要排查'
if __name__ == '__main__':
    main()