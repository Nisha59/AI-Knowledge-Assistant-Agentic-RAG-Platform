from dotenv import load_dotenv
from agents.main_agent import MainAgent

load_dotenv()


def main():
    agent = MainAgent()
    result = agent.run("What information do you have in your knowledge base?")
    print(result)


if __name__ == "__main__":
    main()
