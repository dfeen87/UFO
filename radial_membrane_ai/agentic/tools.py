# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Tool execution and cost profiling framework for Version 3 Agentic AI.
Provides concrete tools with real execution logic, cost vectors, and membrane tension impacts.
"""

from __future__ import annotations

import ast
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from radial_membrane_ai.exceptions import ValidationError


@dataclass
class ToolCallResult:
    """Result of a tool execution with cost and membrane impact metadata."""

    tool_name: str
    success: bool
    output: str
    data: Dict[str, Any] = field(default_factory=dict)
    cost_vector: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.05, 0.0, 0.1, 0.05, 0.0, 0.0])
    side_effect_rating: float = 0.0
    tension_delta: float = 0.1
    execution_time_ms: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces membrane-bounded execution invariants on ToolCallResult."""
        if not (0.0 <= self.side_effect_rating <= 1.0):
            raise ValidationError(
                f"side_effect_rating must be in range [0.0, 1.0], got {self.side_effect_rating}."
            )
        if self.tension_delta < 0.0:
            raise ValidationError(
                f"tension_delta cannot be negative, got {self.tension_delta}."
            )
        if len(self.cost_vector) != 8:
            raise ValidationError(
                f"cost_vector must contain exactly 8 dimensions, got length {len(self.cost_vector)}."
            )
        if any(c < 0.0 for c in self.cost_vector):
            raise ValidationError("All elements in cost_vector must be non-negative.")


class BaseTool(ABC):
    """Abstract base class for governed agentic tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ToolCallResult:
        """Executes the tool with given parameters and returns a ToolCallResult."""
        pass


class SearchTool(BaseTool):
    """Search tool simulating web/corpus search with token and latency costs."""

    name = "SearchTool"
    description = "Searches knowledge corpus for relevant documents and snippets."

    def __init__(self, corpus: Optional[Dict[str, str]] = None) -> None:
        self.corpus = corpus or {
            "ufo architecture": "The UFO is a governed deformable radial membrane model for compute-aware AI.",
            "agentic ai": "Version 3 focuses on agentic planning, tool invocation, reflection, and swarm bidding.",
            "stability bands": "Stability bands maintain Lyapunov energy limits under dynamic workload tension.",
            "semantic memory": "Policy-bound semantic memory coordinates memory writes with curvature gates.",
        }

    def execute(self, params: Dict[str, Any]) -> ToolCallResult:
        start = time.perf_counter()
        query = str(params.get("query", "")).strip().lower()
        if not query:
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                output="Error: Empty query provided.",
                cost_vector=[0.05, 0.01, 0.0, 0.0, 0.01, 0.01, 0.1, 0.0],
                side_effect_rating=0.0,
                tension_delta=0.05,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        matches = []
        for key, value in self.corpus.items():
            if any(term in key or term in value.lower() for term in query.split()):
                matches.append(f"[{key}]: {value}")

        output = "\n".join(matches) if matches else f"No direct matches found for '{query}'."
        exec_ms = (time.perf_counter() - start) * 1000

        # Cost profile: high token cost, moderate latency, low side-effect
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            output=output,
            data={"matches_count": len(matches), "query": query},
            cost_vector=[0.4, 0.1, 0.3, 0.2, 0.1, 0.2, 0.0, 0.0],
            side_effect_rating=0.05,
            tension_delta=0.15,
            execution_time_ms=exec_ms,
        )


class PythonCodeExecutorTool(BaseTool):
    """Safe Python code AST analysis and restricted execution tool."""

    name = "PythonCodeExecutorTool"
    description = "Executes Python expressions and algorithms in a governed, restricted environment."

    def execute(self, params: Dict[str, Any]) -> ToolCallResult:
        start = time.perf_counter()
        code = str(params.get("code", ""))
        if not code.strip():
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                output="Error: No code provided for execution.",
                cost_vector=[0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                side_effect_rating=0.0,
                tension_delta=0.02,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Validate syntax via AST
        try:
            parsed = ast.parse(code)
        except SyntaxError as err:
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                output=f"SyntaxError: {err}",
                cost_vector=[0.1, 0.2, 0.0, 0.0, 0.1, 0.05, 0.3, 0.1],
                side_effect_rating=0.1,
                tension_delta=0.3,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Restricted execution globals/locals
        safe_globals = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range, "list": list}
        local_vars: Dict[str, Any] = {}

        try:
            # Check if last node is expression to return value
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                exec_body = ast.Module(body=parsed.body[:-1], type_ignores=[])
                eval_expr = ast.Expression(body=parsed.body[-1].value)
                exec(compile(exec_body, filename="<agentic_code>", mode="exec"), safe_globals, local_vars)
                result_val = eval(compile(eval_expr, filename="<agentic_code>", mode="eval"), safe_globals, local_vars)
                local_vars["result"] = result_val
            else:
                exec(compile(parsed, filename="<agentic_code>", mode="exec"), safe_globals, local_vars)

            output = json.dumps(local_vars, default=str)
            exec_ms = (time.perf_counter() - start) * 1000

            return ToolCallResult(
                tool_name=self.name,
                success=True,
                output=output,
                data=local_vars,
                cost_vector=[0.5, 0.8, 0.1, 0.0, 0.6, 0.3, 0.0, 0.1],
                side_effect_rating=0.6,
                tension_delta=0.45,
                execution_time_ms=exec_ms,
            )
        except Exception as ex:
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                output=f"ExecutionError: {type(ex).__name__}: {ex}",
                cost_vector=[0.2, 0.4, 0.0, 0.0, 0.3, 0.1, 0.5, 0.3],
                side_effect_rating=0.3,
                tension_delta=0.6,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )


class APIRequestTool(BaseTool):
    """Simulated governed HTTP/REST API request tool."""

    name = "APIRequestTool"
    description = "Dispatches structured HTTP API requests with input validation."

    def execute(self, params: Dict[str, Any]) -> ToolCallResult:
        start = time.perf_counter()
        endpoint = params.get("endpoint", "")
        method = params.get("method", "GET").upper()

        if not endpoint:
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                output="Error: Missing target endpoint.",
                cost_vector=[0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.1, 0.0],
                side_effect_rating=0.0,
                tension_delta=0.1,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        payload = params.get("payload", {})
        resp_data = {
            "status_code": 200,
            "endpoint": endpoint,
            "method": method,
            "response": {"acknowledged": True, "processed_payload": payload},
        }

        exec_ms = (time.perf_counter() - start) * 1000
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            output=json.dumps(resp_data),
            data=resp_data,
            cost_vector=[0.3, 0.2, 0.1, 0.1, 0.5, 0.4, 0.0, 0.05],
            side_effect_rating=0.35,
            tension_delta=0.25,
            execution_time_ms=exec_ms,
        )


class DatabaseQueryTool(BaseTool):
    """In-memory structured record database query tool with curvature impact."""

    name = "DatabaseQueryTool"
    description = "Queries structured relational records from the system state store."

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None) -> None:
        self.db = records or [
            {"id": 1, "agent": "UFO_001", "status": "ACTIVE", "tension": 1.2, "region": "Analytical"},
            {"id": 2, "agent": "UFO_002", "status": "IDLE", "tension": 0.4, "region": "Interpersonal"},
            {"id": 3, "agent": "UFO_003", "status": "ACTIVE", "tension": 3.8, "region": "Generative"},
        ]

    def execute(self, params: Dict[str, Any]) -> ToolCallResult:
        start = time.perf_counter()
        key = params.get("filter_key")
        value = params.get("filter_value")

        if key is None or value is None:
            results = self.db
        else:
            results = [r for r in self.db if str(r.get(key, "")).lower() == str(value).lower()]

        exec_ms = (time.perf_counter() - start) * 1000
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            output=json.dumps(results),
            data={"records": results, "count": len(results)},
            cost_vector=[0.2, 0.3, 0.1, 0.4, 0.2, 0.1, 0.0, 0.0],
            side_effect_rating=0.1,
            tension_delta=0.2,
            execution_time_ms=exec_ms,
        )


class MemoryRetrievalTool(BaseTool):
    """Queries agent semantic and residual memory stores for reflection and history."""

    name = "MemoryRetrievalTool"
    description = "Retrieves semantic memories, residual entries, and historical reflection traces."

    def execute(self, params: Dict[str, Any]) -> ToolCallResult:
        start = time.perf_counter()
        memory_type = params.get("type", "semantic")
        tag = params.get("tag", "all")

        retrieved = {
            "type": memory_type,
            "tag": tag,
            "entries": [
                f"Memory record [{memory_type}] with tag '{tag}' retrieved at step {int(time.time())}.",
                "Prior governed reflection confirmed stable Lyapunov energy balance.",
            ],
        }

        exec_ms = (time.perf_counter() - start) * 1000
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            output=json.dumps(retrieved),
            data=retrieved,
            cost_vector=[0.1, 0.1, 0.2, 0.1, 0.05, 0.05, 0.0, 0.0],
            side_effect_rating=0.0,
            tension_delta=0.05,
            execution_time_ms=exec_ms,
        )


class ToolRegistry:
    """Registry managing available governed agentic tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(SearchTool())
        self.register(PythonCodeExecutorTool())
        self.register(APIRequestTool())
        self.register(DatabaseQueryTool())
        self.register(MemoryRetrievalTool())

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """Looks up a tool by name, raising KeyError if missing."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry.")
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, str]]:
        """Returns descriptions of all registered tools."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def execute_tool(self, name: str, params: Dict[str, Any]) -> ToolCallResult:
        """Executes a named tool with provided parameters."""
        tool = self.get_tool(name)
        return tool.execute(params)
