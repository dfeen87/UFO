# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Multi-agent swarm contract bidding and auction negotiation for Version 3 Agentic AI.
Agents bid based on quantitative membrane state, stability margins, and capability roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence
from radial_membrane_ai.agentic.contracts import GovernedRunResult, IntentContract, stable_digest
from radial_membrane_ai.exceptions import ValidationError


@dataclass(frozen=True)
class AgentBid:
    """A bid submitted by a UFO agent for a swarm task contract."""

    agent_id: str
    contract_id: str
    bid_score: float
    estimated_cost: float
    estimated_tension: float
    agent_capabilities: Sequence[str] = field(default_factory=tuple)
    membrane_stability_margin: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_capabilities", tuple(self.agent_capabilities))
        self.validate()

    def validate(self) -> None:
        """Enforces bid metrics invariants."""
        metrics = (self.bid_score, self.estimated_cost, self.estimated_tension, self.membrane_stability_margin)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics):
            raise ValidationError("bid metrics must be finite numeric values.")
        if not self.agent_id or not self.contract_id:
            raise ValidationError("agent_id and contract_id cannot be empty.")
        if self.bid_score < 0.0:
            raise ValidationError(f"bid_score cannot be negative, got {self.bid_score}.")
        if self.estimated_cost < 0.0:
            raise ValidationError(f"estimated_cost cannot be negative, got {self.estimated_cost}.")
        if self.estimated_tension < 0.0:
            raise ValidationError(f"estimated_tension cannot be negative, got {self.estimated_tension}.")
        if self.membrane_stability_margin < 0.0:
            raise ValidationError(
                f"membrane_stability_margin cannot be negative, got {self.membrane_stability_margin}."
            )


@dataclass
class SwarmContract:
    """A contract task requiring multi-agent auction negotiation."""

    contract_id: str
    goal: str
    required_capabilities: List[str] = field(default_factory=list)
    max_budget: float = 10.0
    assigned_agent_id: Optional[str] = None
    status: str = "OPEN"  # OPEN, AWARDED, FAILED, COMPLETED
    winning_bid: Optional[AgentBid] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces swarm contract invariants."""
        if (
            not isinstance(self.max_budget, (int, float))
            or not math.isfinite(self.max_budget)
            or self.max_budget <= 0.0
        ):
            raise ValidationError(f"max_budget must be positive, got {self.max_budget}.")
        if not self.contract_id or not self.goal:
            raise ValidationError("contract_id and goal cannot be empty.")


@dataclass(frozen=True)
class GovernedSwarmTask:
    """Separates auction requirements from explicit execution authority."""

    contract_id: str
    goal: str
    required_capabilities: tuple[str, ...]
    max_selection_budget: float
    intent: IntentContract

    def __post_init__(self) -> None:
        if not self.contract_id or not self.goal:
            raise ValidationError("governed swarm task identity cannot be empty.")
        object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        if not all(isinstance(value, str) and value for value in self.required_capabilities):
            raise ValidationError("selection capabilities must be identifiers.")
        if not math.isfinite(self.max_selection_budget) or self.max_selection_budget <= 0.0:
            raise ValidationError("selection budget must be positive and finite.")
        if not isinstance(self.intent, IntentContract):
            raise ValidationError("governed swarm execution requires an explicit IntentContract.")
        if self.intent.goal != self.goal:
            raise ValidationError("swarm task goal must exactly match its governing intent.")


@dataclass(frozen=True)
class GovernedSwarmResult:
    """Distinct deterministic selection evidence and governed execution evidence."""

    task: GovernedSwarmTask
    selected_agent_id: str
    winning_bid: AgentBid
    admissible_bids: tuple[AgentBid, ...]
    selection_digest: str
    pre_execution_state_fingerprint: str
    execution_binding_digest: str
    governed_run: GovernedRunResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "admissible_bids", tuple(self.admissible_bids))
        if not self.admissible_bids or len({bid.agent_id for bid in self.admissible_bids}) != len(
            self.admissible_bids
        ):
            raise ValidationError("selection requires unique admissible agent bids.")
        for bid in self.admissible_bids:
            if (
                bid.contract_id != self.task.contract_id
                or not set(self.task.required_capabilities).issubset(bid.agent_capabilities)
                or bid.estimated_cost > self.task.max_selection_budget
                or bid.membrane_stability_margin <= 0.0
            ):
                raise ValidationError("selection contains a bid inadmissible for the governed task.")
        deterministic_bids = tuple(sorted(
            self.admissible_bids, key=lambda bid: (-bid.bid_score, bid.agent_id)
        ))
        deterministic_winner = deterministic_bids[0]
        if self.winning_bid != deterministic_winner:
            raise ValidationError("winning bid does not match the deterministic auction winner.")
        if self.winning_bid.agent_id != self.selected_agent_id:
            raise ValidationError("selected agent is not the deterministic auction winner.")
        expected = stable_digest({
            "task": self.task,
            "selected_agent_id": self.selected_agent_id,
            "winning_bid": self.winning_bid,
            "admissible_bids": deterministic_bids,
        })
        if self.selection_digest != expected:
            raise ValidationError("governed swarm selection evidence is forged.")
        if (
            self.governed_run.intent_id != self.task.intent.intent_id
            or self.governed_run.intent_digest != self.task.intent.intent_digest
        ):
            raise ValidationError("governed swarm run is not bound to its explicit intent.")
        if not self.governed_run.receipts:
            raise ValidationError("governed swarm run requires initial execution evidence.")
        initial_receipt = self.governed_run.receipts[0]
        if initial_receipt.initial_state_fingerprint != self.pre_execution_state_fingerprint:
            raise ValidationError("governed run is not bound to the winner's pre-execution state.")
        expected_execution_binding = stable_digest({
            "selection_digest": self.selection_digest,
            "selected_agent_id": self.selected_agent_id,
            "pre_execution_state_fingerprint": self.pre_execution_state_fingerprint,
            "initial_receipt_digest": stable_digest(initial_receipt),
            "intent_digest": self.task.intent.intent_digest,
        })
        if self.execution_binding_digest != expected_execution_binding:
            raise ValidationError("governed swarm execution binding is forged.")


class SwarmAuctioneer:
    """Manages contract bidding, auction evaluation, and swarm role assignments."""

    def __init__(self) -> None:
        self.contracts: Dict[str, SwarmContract] = {}
        self.auction_history: List[Dict[str, Any]] = []

    def create_contract(
        self,
        contract_id: str,
        goal: str,
        required_capabilities: Sequence[str],
        max_budget: float = 10.0,
    ) -> SwarmContract:
        """Creates and registers a new open swarm task contract."""
        if contract_id in self.contracts:
            raise ValidationError(f"Contract '{contract_id}' already exists.")
        contract = SwarmContract(
            contract_id=contract_id,
            goal=goal,
            required_capabilities=list(required_capabilities),
            max_budget=max_budget,
            status="OPEN",
        )
        self.contracts[contract_id] = contract
        return contract

    def calculate_agent_bid(
        self,
        agent_id: str,
        contract: SwarmContract,
        agent_capabilities: Sequence[str],
        current_tension: float,
        stability_band_margin: float,
        cost_weight: float = 0.5,
    ) -> Optional[AgentBid]:
        """Calculates a quantitative bid for an agent based on membrane state and capabilities."""
        if not all(math.isfinite(v) for v in (current_tension, stability_band_margin, cost_weight)):
            raise ValidationError("bid inputs must be finite.")
        if current_tension < 0.0 or cost_weight < 0.0:
            raise ValidationError("current_tension and cost_weight cannot be negative.")
        # Capability overlap check
        matched = [cap for cap in contract.required_capabilities if cap in agent_capabilities]
        if not matched and contract.required_capabilities:
            return None  # Ineligible bid

        cap_ratio = len(matched) / max(1, len(contract.required_capabilities))

        # Base estimated cost and tension
        est_cost = (1.0 - cap_ratio * 0.5) * 2.0
        est_tension = current_tension + 0.2

        if est_cost > contract.max_budget or stability_band_margin <= 0.0:
            return None  # Exceeds contract budget or unstable

        # Higher score is better: favors capability ratio & stability margin, penalizes high tension & cost
        bid_score = (
            (cap_ratio * 4.0) + (stability_band_margin * 3.0) - (current_tension * 0.5) - (est_cost * cost_weight)
        )

        return AgentBid(
            agent_id=agent_id,
            contract_id=contract.contract_id,
            bid_score=max(0.01, bid_score),
            estimated_cost=est_cost,
            estimated_tension=est_tension,
            agent_capabilities=list(agent_capabilities),
            membrane_stability_margin=stability_band_margin,
        )

    def run_auction(
        self,
        contract_id: str,
        bids: Sequence[AgentBid],
    ) -> SwarmContract:
        """Evaluates bids and awards the contract to the highest-scoring admissible agent."""
        if contract_id not in self.contracts:
            raise KeyError(f"Contract '{contract_id}' not found.")

        contract = self.contracts[contract_id]
        if not bids:
            contract.status = "FAILED"
            return contract

        # Filter admissible bids (cost <= budget, positive stability margin) and sort deterministically
        admissible_bids = [
            b for b in bids
            if b.contract_id == contract_id
            and set(contract.required_capabilities).issubset(set(b.agent_capabilities))
            and b.estimated_cost <= contract.max_budget
            and b.membrane_stability_margin > 0.0
        ]
        if not admissible_bids:
            contract.status = "FAILED"
            return contract

        sorted_bids = sorted(admissible_bids, key=lambda b: (-b.bid_score, b.agent_id))
        winning_bid = sorted_bids[0]

        contract.assigned_agent_id = winning_bid.agent_id
        contract.winning_bid = winning_bid
        contract.status = "AWARDED"

        self.auction_history.append({
            "contract_id": contract_id,
            "winner": winning_bid.agent_id,
            "bid_score": winning_bid.bid_score,
            "bids_count": len(bids),
        })

        return contract
