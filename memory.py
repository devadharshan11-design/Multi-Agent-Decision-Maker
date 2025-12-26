class SimpleMemory:
    """
    Very simple in-RAM memory. Later we can replace this
    with a vector database if needed.
    """
    def __init__(self):
        self.logs = []

    def add(self, agent_name: str, content: str):
        self.logs.append({"agent": agent_name, "content": content})

    def dump_as_text(self) -> str:
        if not self.logs:
            return "(no prior memory)"
        return "\n".join(f"[{x['agent']}] {x['content']}" for x in self.logs)
