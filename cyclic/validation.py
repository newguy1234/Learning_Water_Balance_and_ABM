"""Training-year selection and one final evaluation on a distinct held-out year."""
import copy
import time

import pandas as pd


def output_label(policies, year):
    """Viewing any year is allowed; distinguish diagnostics from an unseen year."""
    agents = list(policies.values()) if isinstance(policies, dict) else [policies]
    histories = []
    for agent in agents:
        history = getattr(agent, "training_years", ())
        # Package et_net stores train years in et_net.pt; stamp before model.reset().
        if (not history) and getattr(agent, "name", None) == "et_net":
            try:
                from .et_net import weights_status
                status = weights_status()
                if status.compatible:
                    history = tuple(status.train_years)
                    agent.training_years = history
            except Exception:
                pass
        histories.append(history)
    if any(history is not None and year in history for history in histories):
        return "training-year diagnostic"
    if any(history is None for history in histories):
        return "diagnostic — training history unknown"
    if any(histories):
        return "unseen-year output (learning off)"
    return "untrained-policy output"


def split(year=2021, evaluation_year=None, observed=None, policies=None):
    year = int(year)
    evaluation_year = (2022 if year == 2021 else 2021) if evaluation_year is None else int(evaluation_year)
    if year == evaluation_year:
        raise ValueError("Training and evaluation years must differ")
    if year not in (2021, 2022) or evaluation_year not in (2021, 2022):
        raise ValueError("Supported training/evaluation years are 2021 and 2022")
    if observed is not None:
        for frame in observed[:2]:
            if frame is not None and not frame.empty and "date" in frame:
                if not (pd.to_datetime(frame["date"]).dt.year == year).all():
                    raise ValueError("Training observations contain dates outside the training year")
    if policies is not None:
        agents = policies.values() if isinstance(policies, dict) else [policies]
        for agent in agents:
            history = getattr(agent, "training_years", ())
            if history is None:
                raise ValueError("This saved agent has unknown training years; start from a fresh registered agent")
            if evaluation_year in history:
                raise ValueError(f"Agent already trained on evaluation year {evaluation_year}; start from a fresh agent")
    return year, evaluation_year


def evaluate(result, year, evaluation_year=None, physics=None):
    """Freeze the selected training snapshot, then evaluate without learning hooks.

    Held-out scores never participate in candidate selection or training history.
    Returned policies retain the pre-evaluation snapshot and its provenance.
    """
    from . import flyover, imitation
    from .model import CyclicModel

    started = time.time()
    year, evaluation_year = split(year, evaluation_year, policies=result.policies)
    result.policies = copy.deepcopy(result.policies)
    for policy in result.policies.values():
        policy.training_years = tuple(sorted(set(getattr(policy, "training_years", ())) | {year}))
    result.notes.update(training_year=year, evaluation_year=evaluation_year, training_score=result.score)
    result.history = result.history.assign(training_year=year)
    if "agent" in result.notes:
        result.notes["agent"] = next(iter(result.policies.values()))
    held_out = CyclicModel(evaluation_year, agents=copy.deepcopy(result.policies), physics=physics).run()
    result.results = held_out
    result.score = imitation.score(held_out.blocks, imitation.load_observed(evaluation_year)[0])
    result.score.flyover = flyover.from_results(held_out)
    result.score.et_flyover_fidelity = result.score.flyover.fidelity
    result.notes["evaluation_learning_enabled"] = False
    result.seconds += time.time() - started
    return result
