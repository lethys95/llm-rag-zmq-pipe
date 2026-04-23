from dataclasses import dataclass


@dataclass
class AdvisorOutput:
    """Natural language guidance produced by an advisor node.

    Advisors consume classifier outputs and translate them into language
    the primary LLM can act on. The primary LLM never sees raw scores.

    potency — how relevant is this advice to the current moment (0.0–1.0).
              Not confidence. An advisor can be certain and still have low
              potency because this moment doesn't call for that kind of guidance.
    """

    advisor: str        # name, for transparency and debugging
    advice: str         # natural language guidance
    potency: float      # 0.0–1.0, relevance to this moment
