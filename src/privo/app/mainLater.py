from app.config import load_config
from app.orchestrator import AssistantOrchestrator


def main():
    config = load_config()
    assistant = AssistantOrchestrator(config)
    assistant.process_once()


if __name__ == "__main__":
    main()