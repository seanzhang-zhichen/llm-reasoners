from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_fireworks import ChatFireworks
import dotenv


dotenv.load_dotenv(override=True)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名作文助手，负责撰写优质的五段式作文。"
            " 请根据用户的请求生成尽可能优秀的文章。"
            " 如果用户提供批评意见，请基于你之前的尝试给出修订版本。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatFireworks(
    model="accounts/fireworks/models/glm-4p6", max_tokens=25344
)
generate = prompt | llm


essay = ""
request = HumanMessage(
    content="写一篇关于《小王子》为何与现代童年相关的文章。"
)
for chunk in generate.stream({"messages": [request]}):
    print(chunk.content, end="")
    essay += chunk.content
    
print()
print("===加入老师评价===")

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名教师，负责对学生的作文进行评估。"
            " 请根据学生的作文内容，生成对其的批评和建议。"
            " 建议包括长度、深度、风格等方面的改进。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

reflection = ""
for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
    print(chunk.content, end="")
    reflection += chunk.content


print()
print("===加入老师评价后的作文===")
    
for chunk in generate.stream(
    {"messages": [request, AIMessage(content=essay), HumanMessage(content=reflection)]}
):
    print(chunk.content, end="")