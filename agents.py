from llm_client import call_llm


class BaseAgent:
    def __init__(self, name, role, memory, mode="general"):
        self.name = name
        self.role = role
        self.memory = memory
        self.mode = mode  # "engineering" / "policy" / "research" / "general"

    def run(self, task, context: str = "") -> str:
        system_prompt = (
            f"You are {self.name}, {self.role}. "
            "Provide structured, technical responses when appropriate."
        )

        full_prompt = (
            f"Task:\n{task}\n\n"
            f"Context:\n{context}\n\n"
            f"Recent system memory:\n{self.memory.dump_as_text()}"
        )

        result = call_llm(system_prompt, full_prompt)
        self.memory.add(self.name, result)
        return result


class PlannerAgent(BaseAgent):
    """
    Breaks a high-level goal into executable substeps.
    """

    def run(self, task, context: str = "") -> str:

        if self.mode == "engineering":
            style_hint = (
                "Focus on system design, algorithms, technologies, constraints, and implementation details."
            )
        elif self.mode == "policy":
            style_hint = (
                "Focus on regulations, governance models, societal impact, and deployment strategy."
            )
        elif self.mode == "research":
            style_hint = (
                "Focus on academic style: literature review, hypothesis, methodology, experiments."
            )
        else:
            style_hint = "Use a balanced, general planning style."

        system_prompt = (
            "You are a planning agent. "
            f"MODE = {self.mode}. {style_hint} "
            "Break the user goal into numbered, clear, practical steps."
        )

        full_prompt = (
            f"User Goal:\n{task}\n\n"
            f"Additional Context:\n{context}\n\n"
            f"Memory:\n{self.memory.dump_as_text()}"
        )

        result = call_llm(system_prompt, full_prompt)
        self.memory.add(self.name, result)
        return result


class SolverAgent(BaseAgent):
    """
    Converts plan into a real solution.
    """

    def run(self, task, context: str = "") -> str:

        if self.mode == "engineering":
            style_hint = "Provide architecture, tools, algorithms and implementation steps."
        elif self.mode == "policy":
            style_hint = "Provide governance, policy frameworks, rollout strategy, and risk analysis."
        elif self.mode == "research":
            style_hint = "Provide a research-style output: method, evaluation, experiments, contributions."
        else:
            style_hint = "Provide a structured general solution."

        system_prompt = (
            "You are a solution-generation agent. "
            f"MODE = {self.mode}. {style_hint} "
            "Use the plan to generate a complete and structured answer."
        )

        full_prompt = (
            f"Task:\n{task}\n\n"
            f"Plan:\n{context}\n\n"
            f"Memory:\n{self.memory.dump_as_text()}"
        )

        result = call_llm(system_prompt, full_prompt)
        self.memory.add(self.name, result)
        return result


class ImprovementAgent(BaseAgent):
    """
    Improves solution using evaluator feedback.
    """

    def run(self, task, context: str = "") -> str:

        system_prompt = (
            "You are an expert AI improvement agent.\n"
            "You receive: original solution + critique.\n\n"
            "Your instructions:\n"
            "1. Fix weaknesses mentioned\n"
            "2. Reduce unnecessary content\n"
            "3. Improve feasibility\n"
            "4. Increase clarity\n"
            "5. Improve technical depth\n"
            "6. Keep concise and professional\n\n"
            "Return an improved final solution."
        )

        full_prompt = (
            f"{task}\n\n"
            f"Context:\n{context}\n\n"
            f"Memory:\n{self.memory.dump_as_text()}"
        )

        result = call_llm(system_prompt, full_prompt)
        self.memory.add(self.name, result)
        return result
