"""Validated, redacted, dry-run-first RunPod Pod lifecycle plans."""

from __future__ import annotations

import json
import os
import re
from contextlib import suppress
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cadence.common.config import CadenceConfig
from cadence.common.repro import stable_hash

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
RunPodAction = Literal["search", "create", "inspect", "terminate"]
HttpMethod = Literal["GET", "POST", "DELETE"]


class RunPodPlan(BaseModel):
    """Public-safe representation of one future RunPod API request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"] = "0.1.0"
    provider: Literal["runpod"] = "runpod"
    action: RunPodAction
    method: HttpMethod
    endpoint: str
    request_body: dict[str, Any] | None = None
    credential_environment_variable: Literal["RUNPOD_API_KEY"] = "RUNPOD_API_KEY"
    requires_explicit_execute: Literal[True] = True
    requires_human_approval: bool
    destructive: bool
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_hardware: Literal["NVIDIA RTX A5000 24GB"]
    maximum_hourly_price_usd: float = Field(gt=0)
    maximum_budget_usd: float = Field(gt=0, le=5)
    maximum_runtime_minutes: int = Field(gt=0, le=240)
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def validate_public_safe_request(self) -> RunPodPlan:
        if not self.endpoint.startswith(f"{RUNPOD_API_BASE}/pods"):
            raise ValueError("RunPod plans may only target the official Pods API")
        encoded = json.dumps(self.request_body, sort_keys=True).lower()
        for forbidden in ("api_key", "authorization", "token", "secret", "manifest", "checkpoint"):
            if forbidden in encoded:
                raise ValueError(f"RunPod request body contains forbidden field: {forbidden}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_runpod_plan(
    action: RunPodAction,
    config: CadenceConfig,
    *,
    pod_id: str | None = None,
) -> RunPodPlan:
    """Build a deterministic plan without reading credentials or contacting RunPod."""
    remote = config.remote
    if remote.provider != "runpod":
        raise ValueError("RunPod actions require remote.provider=runpod")

    body: dict[str, Any] | None = None
    if action == "search":
        method: HttpMethod = "GET"
        endpoint = f"{RUNPOD_API_BASE}/pods"
    elif action == "create":
        method = "POST"
        endpoint = f"{RUNPOD_API_BASE}/pods"
        body = {
            "allowedCudaVersions": ["12.6"],
            "cloudType": remote.runpod_cloud_type,
            "computeType": "GPU",
            "containerDiskInGb": remote.runpod_container_disk_gb,
            "dockerEntrypoint": [],
            "dockerStartCmd": [],
            "gpuCount": remote.runpod_gpu_count,
            "gpuTypeIds": [remote.runpod_gpu_type_id],
            "gpuTypePriority": "custom",
            "imageName": remote.runpod_image_name,
            "interruptible": False,
            "locked": False,
            "name": remote.runpod_pod_name,
            "ports": [],
            "supportPublicIp": False,
            "volumeInGb": remote.runpod_volume_gb,
            "volumeMountPath": "/workspace",
        }
    else:
        validated_pod_id = _validate_pod_id(pod_id)
        endpoint = f"{RUNPOD_API_BASE}/pods/{validated_pod_id}"
        method = "GET" if action == "inspect" else "DELETE"

    requires_approval = action in {"create", "terminate"}
    plan_seed = {
        "action": action,
        "method": method,
        "endpoint": endpoint,
        "request_body": body,
        "requested_hardware": remote.requested_hardware,
        "maximum_hourly_price_usd": remote.maximum_hourly_price_usd,
        "maximum_budget_usd": remote.maximum_budget_usd,
        "maximum_runtime_minutes": remote.maximum_runtime_minutes,
    }
    return RunPodPlan(
        action=action,
        method=method,
        endpoint=endpoint,
        request_body=body,
        requires_human_approval=requires_approval,
        destructive=action == "terminate",
        plan_hash=stable_hash(plan_seed),
        requested_hardware="NVIDIA RTX A5000 24GB",
        maximum_hourly_price_usd=remote.maximum_hourly_price_usd,
        maximum_budget_usd=remote.maximum_budget_usd,
        maximum_runtime_minutes=remote.maximum_runtime_minutes,
        warnings=(
            "A dry run is not authorization to provision hardware or spend.",
            "Stopping a Pod can retain billable volume storage; terminate unused Pods.",
            "Termination deletes non-network-volume data; export approved artifacts first.",
            "Verify termination and persistent storage separately after every executed job.",
        ),
    )


def execute_runpod_plan(
    plan: RunPodPlan,
    *,
    execute: bool,
    approval_reference: str | None = None,
    confirm_termination: bool = False,
) -> dict[str, Any]:
    """Execute only after explicit CLI and human gates; return sanitized results."""
    if not execute:
        return {"mode": "dry-run", "network_action": False, "plan": plan.to_dict()}

    api_key = os.getenv(RUNPOD_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{RUNPOD_API_KEY_ENV} is required for an executed RunPod action")
    if plan.requires_human_approval and not _valid_approval_reference(approval_reference):
        raise ValueError("create and terminate require an opaque human approval reference")
    if plan.action == "terminate" and not confirm_termination:
        raise ValueError("termination requires --confirm-termination")

    if plan.action == "create":
        if plan.request_body is None:
            raise RuntimeError("RunPod create plan is missing its request body")
        existing = _request_json("GET", f"{RUNPOD_API_BASE}/pods", api_key, None)
        if _contains_named_pod(existing, str(plan.request_body["name"])):
            return _result(plan, status="already-exists", network_action=True)

    try:
        response = _request_json(plan.method, plan.endpoint, api_key, plan.request_body)
    except HTTPError as exc:
        if plan.action == "terminate" and exc.code == 404:
            return _result(plan, status="already-terminated", network_action=True)
        raise RuntimeError(f"RunPod API request failed with HTTP {exc.code}") from None

    if plan.action == "create":
        _enforce_created_price(plan, response, api_key)
        return _result(plan, status="created", network_action=True)
    if plan.action == "terminate":
        verified = _termination_is_verified(plan.endpoint, api_key)
        if not verified:
            raise RuntimeError("RunPod termination could not be verified")
        return _result(plan, status="terminated-and-verified", network_action=True)
    if plan.action == "search":
        return {
            **_result(plan, status="inspected", network_action=True),
            "pod_count": _pod_count(response),
        }
    return {
        **_result(plan, status="inspected", network_action=True),
        "desired_status": _safe_status(response),
    }


def _validate_pod_id(pod_id: str | None) -> str:
    if pod_id is None or re.fullmatch(r"[A-Za-z0-9_-]{3,128}", pod_id) is None:
        raise ValueError("inspect and terminate require a valid RunPod Pod ID")
    return pod_id


def _valid_approval_reference(reference: str | None) -> bool:
    return (
        reference is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", reference) is not None
    )


def _request_json(
    method: HttpMethod,
    endpoint: str,
    api_key: str,
    body: dict[str, Any] | None,
) -> Any:
    encoded = None if body is None else json.dumps(body, sort_keys=True).encode()
    request = Request(
        endpoint,
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
    except HTTPError:
        raise
    if not raw:
        return {}
    return json.loads(raw)


def _contains_named_pod(response: Any, name: str) -> bool:
    pods = response if isinstance(response, list) else response.get("pods", [])
    return isinstance(pods, list) and any(
        isinstance(pod, dict) and pod.get("name") == name for pod in pods
    )


def _enforce_created_price(plan: RunPodPlan, response: Any, api_key: str) -> None:
    if not isinstance(response, dict):
        raise RuntimeError("RunPod create response was not an object")
    raw_cost = response.get("costPerHr", response.get("adjustedCostPerHr"))
    if not isinstance(raw_cost, (int, float, str)):
        raise RuntimeError("RunPod create response did not include an hourly price")
    try:
        cost = float(raw_cost)
    except (TypeError, ValueError):
        raise RuntimeError("RunPod create response did not include an hourly price") from None
    if cost <= plan.maximum_hourly_price_usd:
        return
    pod_id = response.get("id")
    if isinstance(pod_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{3,128}", pod_id):
        with suppress(HTTPError):
            _request_json("DELETE", f"{RUNPOD_API_BASE}/pods/{pod_id}", api_key, None)
    raise RuntimeError(
        "created Pod exceeded the approved hourly price and termination was attempted"
    )


def _termination_is_verified(endpoint: str, api_key: str) -> bool:
    try:
        _request_json("GET", endpoint, api_key, None)
    except HTTPError as exc:
        return exc.code == 404
    return False


def _pod_count(response: Any) -> int:
    pods = response if isinstance(response, list) else response.get("pods", [])
    return len(pods) if isinstance(pods, list) else 0


def _safe_status(response: Any) -> str:
    if not isinstance(response, dict):
        return "unknown"
    status = response.get("desiredStatus")
    return status if isinstance(status, str) else "unknown"


def _result(plan: RunPodPlan, *, status: str, network_action: bool) -> dict[str, Any]:
    return {
        "action": plan.action,
        "network_action": network_action,
        "plan_hash": plan.plan_hash,
        "provider": plan.provider,
        "status": status,
    }
