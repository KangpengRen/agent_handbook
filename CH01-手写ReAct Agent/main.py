# main.py
from agent import ReActAgent
from tools import create_default_tools


def main():
    # 创建工具
    tools = create_default_tools()

    # 创建 Agent
    agent = ReActAgent(tools, model="gpt-4o-mini")

    # 测试用例
    test_cases = [
        "北京今天天气怎么样？",
        "帮我计算 (123 + 456) * 789 的结果",
        "今天是几号？北京天气如何？",
    ]

    for query in test_cases:
        result = agent.run(query)
        print(f"\n📝 回答: {result}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()