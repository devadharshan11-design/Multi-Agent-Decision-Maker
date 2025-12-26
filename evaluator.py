from llm_client import call_llm


class EvaluatorAgent:
    """
    Evaluates answers from other agents and gives a structured score.
    """

    def __init__(self, memory):
        self.memory = memory
        self.name = "Evaluator"

    def evaluate(self, original_task: str, candidate_answer: str) -> str:
        system_prompt = (
            "You are an AI evaluator. "
            "Given a task and an answer, you must:\n"
            "1) Summarize the answer in 2–3 lines\n"
            "2) List strengths\n"
            "3) List weaknesses\n"
            "4) Give a numerical score from 0 to 10\n"
            "5) Suggest concrete improvements.\n\n"
            "VERY IMPORTANT: At the end, output a separate final line in the format:\n"
            "SCORE: x.y\n"
            "Where x.y is just the numeric score (e.g., SCORE: 7.5)."
        )

        user_prompt = (
            f"Task:\n{original_task}\n\n"
            f"Candidate answer:\n{candidate_answer}\n"
        )

        result = call_llm(system_prompt, user_prompt)
        self.memory.add(self.name, result)
        return result
