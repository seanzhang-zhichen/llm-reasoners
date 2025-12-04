from langchain_classic import hub
import os
import dotenv

dotenv.load_dotenv(override=True)

select_prompt = hub.pull("hwchase17/self-discovery-select")
print("Self-Discovery Select Prompt:")
select_prompt.pretty_print()
print("====="*10)

print("Self-Discovery Select Response:")
adapt_prompt = hub.pull("hwchase17/self-discovery-adapt")
adapt_prompt.pretty_print()
print("====="*10)

structured_prompt = hub.pull("hwchase17/self-discovery-structure")
print("Self-Discovery Structured Prompt:")
structured_prompt.pretty_print()
print("====="*10)

reasoning_prompt = hub.pull("hwchase17/self-discovery-reasoning")
print("Self-Discovery Structured Response:")
reasoning_prompt.pretty_print()
print("====="*10)




from typing import Optional
from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_fireworks import ChatFireworks
from langgraph.graph import END, START, StateGraph


class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]


model = ChatOpenAI(
    model="gpt-4o", max_tokens=16384, streaming=True, base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
)



def select(inputs):
    select_chain = select_prompt | model | StrOutputParser()
    return {"selected_modules": select_chain.invoke(inputs)}


def adapt(inputs):
    adapt_chain = adapt_prompt | model | StrOutputParser()
    return {"adapted_modules": adapt_chain.invoke(inputs)}


def structure(inputs):
    structure_chain = structured_prompt | model | StrOutputParser()
    return {"reasoning_structure": structure_chain.invoke(inputs)}


def reason(inputs):
    reasoning_chain = reasoning_prompt | model | StrOutputParser()
    return {"answer": reasoning_chain.invoke(inputs)}


graph = StateGraph(SelfDiscoverState)
graph.add_node(select)
graph.add_node(adapt)
graph.add_node(structure)
graph.add_node(reason)
graph.add_edge(START, "select")
graph.add_edge("select", "adapt")
graph.add_edge("adapt", "structure")
graph.add_edge("structure", "reason")
graph.add_edge("reason", END)
app = graph.compile()



reasoning_modules = [
    "1. 我如何设计一个实验来帮助解决这个问题？",
    "2. 列出解决这个问题的想法清单，并逐一应用到问题上，看看是否能取得进展。",
    # "3. 我如何衡量这个问题的进展？",
    "4. 我如何简化这个问题，使其更容易解决？",
    "5. 这个问题的关键假设是什么？",
    "6. 每个解决方案的潜在风险和缺点是什么？",
    "7. 对于这个问题，有哪些替代的观点或视角？",
    "8. 这个问题及其解决方案的长期影响是什么？",
    "9. 我如何将这个问题分解成更小、更易管理的部分？",
    "10. 批判性思维：这种方式涉及从不同角度分析问题，质疑假设，并评估可用的证据或信息。它专注于逻辑推理、基于证据的决策，以及识别思维中的潜在偏见或缺陷。",
    "11. 尝试创造性思维，产生创新和突破常规的想法来解决问题。探索非传统的解决方案，超越传统边界思考，并鼓励想象力和原创性。",
    # "12. 寻求他人的意见和协作来解决问题。强调团队合作、开放沟通，以及利用团队的多元视角和专业知识来提出有效的解决方案。",
    "13. 使用系统思维：将问题视为更大系统的一部分，理解各要素之间的相互联系。专注于识别影响问题的根本原因、反馈循环和相互依赖关系，并开发能够整体解决系统问题的全面解决方案。",
    "14. 使用风险分析：评估与不同解决方案或方法相关的潜在风险、不确定性和权衡。强调评估潜在后果和成功或失败的可能性，并基于风险和收益的平衡分析做出明智决策。",
    # "15. 使用反思性思维：退后一步，花时间进行内省和自我反思。审视可能影响问题解决的个人偏见、假设和心智模型，并乐于从过去的经验中学习以改进未来的方法。",
    "16. 需要解决的核心问题是什么？",
    "17. 导致这个问题的根本原因或因素是什么？",
    "18. 是否有之前尝试过的潜在解决方案或策略？如果有，结果如何，学到了什么教训？",
    "19. 解决这个问题可能会遇到哪些潜在的障碍或挑战？",
    "20. 是否有相关的数据或信息可以提供对问题的洞察？如果有，有哪些数据源可用，如何分析它们？",
    "21. 是否有直接受问题影响的利益相关者或个人？他们的观点和需求是什么？",
    "22. 有效解决这个问题需要哪些资源（财务、人力、技术等）？",
    "23. 如何衡量或评估解决问题的进展或成功？",
    "24. 可以使用哪些指标或度量标准？",
    "25. 这个问题是需要特定专业知识或技能的技术性或实践性问题吗？还是更多是概念性或理论性问题？",
    "26. 这个问题是否涉及物理约束，如有限的资源、基础设施或空间？",
    "27. 这个问题是否与人类行为相关，如社会、文化或心理问题？",
    "28. 这个问题是否涉及决策或规划，需要在不确定性或竞争目标下做出选择？",
    "29. 这是一个需要数据分析、建模或优化技术的分析性问题吗？",
    "30. 这是一个需要创造性解决方案和创新的设计挑战吗？",
    "31. 这个问题是否需要解决系统性或结构性问题，而不仅仅是个别实例？",
    "32. 这个问题是否具有时效性或紧迫性，需要立即关注和行动？",
    "33. 对于这类问题规格，通常会产生什么样的解决方案？",
    "34. 根据问题规格和当前最佳解决方案，猜测其他可能的解决方案。"
    "35. 让我们假设当前最佳解决方案完全错误，还有什么其他方式来思考问题规格？"
    "36. 根据你对这类问题规格的了解，修改当前最佳解决方案的最佳方式是什么？"
    "37. 忽略当前最佳解决方案，为问题创建一个全新的解决方案。"
    # "38. 让我们一步一步思考。"
    "39. 让我们制定一个分步计划，并用良好的符号和解释来实施它。",
]


task_example = "丽莎有10个苹果。她给了朋友3个苹果，然后又从商店买了5个苹果。丽莎现在有多少个苹果？"

# task_example = """这个SVG路径元素 <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
# 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> 绘制的是：
# (A) 圆形 (B) 七边形 (C) 六边形 (D) 风筝形 (E) 直线 (F) 八边形 (G) 五边形 (H) 矩形 (I) 扇形 (J) 三角形"""

reasoning_modules_str = "\n".join(reasoning_modules)

for s in app.stream(
    {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
):
    print(s)