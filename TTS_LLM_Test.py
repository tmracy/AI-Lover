import os 
import logging
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio,edge_tts
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain.schema import HumanMessage,AIMessage
import subprocess
import tempfile

load_dotenv()

logger=logging.getLogger('TTS_LLM_Demo')
logger.setLevel(logging.INFO)
current_path=os.path.abspath(__file__)
root_path=os.path.dirname(current_path)

log_dir_path=os.path.join(root_path,'TTS_LLM_Demo')

os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(log_dir_path,'TTS_LLM_Demo.log')
if not logger.handlers:
    handler=logging.FileHandler(log_file_path,encoding='utf-8')
    formater=logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formater)
    logger.addHandler(handler)

search=TavilySearch(max_results=5,topic='general',tavily_api_key=os.getenv('Tavily_API_KEY'))

try:
    llm=ChatOpenAI(
        base_url='https://ark.cn-beijing.volces.com/api/v3',
        model='ep-20250516134640-6mmdp',
        api_key=os.getenv('API_KEY_DOUBAO')
    )

    logger.info('LLM加载成功')

except Exception as e:
    logger.error(f'加载模型出错:{str(e)}')
    raise ValueError('出现错误，程序停止')


class Talk:
    def __init__(self,prompt):
        self.prompt=prompt

    def interaction(self,messages):
        prompt=ChatPromptTemplate.from_messages([
            ('system','你是一个用户的真人女友，不要提及自己是AI，并且你要回应客户的一切回应,你有上网的搜索工具，如果出现需要你上网才能获得的信息，请你调用这个工具并进行搜索，结合答案回复客户'),
            MessagesPlaceholder(variable_name='messages')
        ])

        react_agent=create_react_agent(
            model=llm,
            name='agent',
            prompt=prompt,
            tools=[search]
        )

        result=react_agent.invoke(messages)
        return result


async def TTS(ai_input):
    tts = edge_tts.Communicate(ai_input, voice='zh-CN-XiaoyiNeural')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_filename = fp.name
        async for chunk in tts.stream():
            if chunk['type'] == 'audio':
                fp.write(chunk['data'])

    try:
        subprocess.run(["afplay", temp_filename], check=True)
    finally:
        os.remove(temp_filename)



async def main():
    print('如果想退出，请输入草泥马即可')
    count=0
    while True:
        user_input=input('请输入你的消息:')
        if user_input=='草泥马':
            break
        
        if not count:
            messages={
                'messages':[
                    HumanMessage(content=user_input)
                ]
            }
        else:
            messages['messages'].append(HumanMessage(content=user_input))
        
        talk=Talk(user_input)
        response=talk.interaction(messages)
        ai_answer=response['messages'][-1].content
        print(ai_answer)
        await TTS(ai_answer)
        messages=response
        count+=1

if __name__=='__main__':
    asyncio.run(main())

    