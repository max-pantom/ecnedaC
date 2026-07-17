"""Private, server-rendered human-review console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from cadence.common.config import CadenceConfig
from cadence.dataset.downloaders import DirectHTTPDownloader, DownloaderChain, YtDlpDownloader
from cadence.dataset.media import FFmpegMediaProcessor
from cadence.dataset.records import ApprovalStatus, RightsStatus, SourceRecord
from cadence.dataset.service import GIB, DatasetIntakeService
from cadence.review.auth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE_SECONDS,
    AuthenticationError,
    LoginAttemptLimiter,
    ReviewSession,
    SessionRole,
    SessionSigner,
    TunnelBasicAuth,
    configured_reviewer_secret,
    configured_reviewer_session_max_age,
    configured_secret,
    verify_administrator_secret,
    verify_csrf,
)
from cadence.review.media import media_response, resolve_registered_media
from cadence.review.models import EvidenceReference, StaleRevisionError

_DIRECTORY = Path(__file__).parent


def create_app(
    config: CadenceConfig,
    service: DatasetIntakeService | None = None,
    auth_secret: str | None = None,
    *,
    secure_cookies: bool = False,
    tunnel_basic_auth: TunnelBasicAuth | None = None,
    allowed_hosts: tuple[str, ...] | None = None,
    login_max_failures: int | None = None,
    reviewer_secret: str | None = None,
    reviewer_session_max_age_seconds: int | None = None,
) -> FastAPI:
    """Build the private console; callers must bind Uvicorn to loopback by default."""
    administrator_secret = configured_secret(auth_secret)
    readonly_secret = configured_reviewer_secret(reviewer_secret)
    readonly_session_max_age = (
        configured_reviewer_session_max_age(reviewer_session_max_age_seconds)
        if readonly_secret is not None
        else 0
    )
    if readonly_secret is not None and verify_administrator_secret(
        readonly_secret, administrator_secret
    ):
        raise ValueError("administrator and read-only secrets must be different")
    signer = SessionSigner(administrator_secret)
    intake = service or _build_service(config)
    templates = Jinja2Templates(directory=_DIRECTORY / "templates")
    login_limiter = (
        LoginAttemptLimiter(maximum_failures=login_max_failures)
        if login_max_failures is not None
        else None
    )

    app = FastAPI(
        title="Cadence private dataset review",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = config
    app.state.intake_service = intake
    app.state.default_bind_host = "127.0.0.1"
    if allowed_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.mount("/static", StaticFiles(directory=_DIRECTORY / "static"), name="static")

    @app.middleware("http")
    async def private_console_headers(request: Request, call_next: Any) -> Response:
        if tunnel_basic_auth is not None and not tunnel_basic_auth.verify(
            request.headers.get("authorization")
        ):
            response = Response(
                "Temporary tunnel authentication required.",
                status_code=401,
                headers={
                    "WWW-Authenticate": 'Basic realm="Cadence temporary review", charset="UTF-8"'
                },
            )
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = response.headers.get(
            "Cache-Control", "private, no-store"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'self'; img-src 'self' data:; media-src 'self'; "
            "script-src 'self'; style-src 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(StaleRevisionError)
    async def stale_revision_handler(request: Request, exc: StaleRevisionError) -> Response:
        payload = {
            "detail": "record changed since this review page was loaded",
            "entity_type": exc.entity_type,
            "entity_id": str(exc.entity_id),
            "expected_revision": exc.expected_revision,
            "actual_revision": exc.actual_revision,
        }
        if _is_json_request(request):
            return JSONResponse(payload, status_code=409)
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": 409, **payload},
            status_code=409,
        )

    @app.exception_handler(KeyError)
    async def unknown_record_handler(request: Request, exc: KeyError) -> Response:
        del exc
        payload = {"detail": "unknown review record"}
        if _is_json_request(request) or request.url.path.startswith("/api/"):
            return JSONResponse(payload, status_code=404)
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": 404, **payload},
            status_code=404,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"readonly_login_enabled": readonly_secret is not None},
        )

    @app.post("/login")
    async def login(request: Request) -> Response:
        if login_limiter is not None and not login_limiter.allowed():
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Too many failed sign-in attempts. Try again in five minutes."},
                status_code=429,
                headers={"Retry-After": "300"},
            )
        payload = await _request_payload(request)
        supplied_secret = _required_text(payload, "secret")
        actor = _required_text(payload, "actor")
        administrator_login = verify_administrator_secret(supplied_secret, administrator_secret)
        reviewer_login = readonly_secret is not None and verify_administrator_secret(
            supplied_secret, readonly_secret
        )
        if not administrator_login and not reviewer_login:
            if login_limiter is not None:
                login_limiter.record_failure()
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "error": "Invalid access secret.",
                    "readonly_login_enabled": readonly_secret is not None,
                },
                status_code=401,
            )
        if login_limiter is not None:
            login_limiter.reset()
        role: SessionRole = "administrator" if administrator_login else "reviewer"
        session_lifetime = (
            SESSION_MAX_AGE_SECONDS if role == "administrator" else readonly_session_max_age
        )
        token, session = signer.issue(
            actor,
            role=role,
            max_age_seconds=session_lifetime,
        )
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=session.expires_at - session.issued_at,
            httponly=True,
            secure=secure_cookies or request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
        return response

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        session, payload = await _authenticated_mutation(
            request, signer, require_administrator=False
        )
        del session, payload
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def review_queue_page(request: Request) -> Response:
        session = _html_session(request, signer)
        if session is None:
            return RedirectResponse("/login", status_code=303)
        queue = intake.review_queue()
        if not session.can_mutate:
            queue = [item for item in queue if item.entity_type == "source"]
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "session": session,
                "csrf_token": session.csrf_token,
                "items": queue,
                "datasets": intake.list_datasets() if session.can_mutate else (),
            },
        )

    @app.get("/datasets/{name}", response_class=HTMLResponse)
    async def dataset_page(request: Request, name: str) -> Response:
        session = _html_session(request, signer)
        if session is None:
            return RedirectResponse("/login", status_code=303)
        _require_administrator(session)
        record = intake.latest_dataset(name)
        return templates.TemplateResponse(
            request,
            "dataset.html",
            {
                "session": session,
                "csrf_token": session.csrf_token,
                "record": record,
                "report": intake.dataset_report(name),
                "events": intake.list_audit_events(
                    entity_type="dataset", entity_id=record.dataset_id
                ),
            },
        )

    @app.get("/sources/{source_id}", response_class=HTMLResponse)
    async def source_page(request: Request, source_id: str) -> Response:
        session = _html_session(request, signer)
        if session is None:
            return RedirectResponse("/login", status_code=303)
        try:
            source = intake.registry.get_source(source_id)
            if session.can_mutate:
                segments = intake.list_segments(source_id)
                events = intake.list_audit_events(entity_type="source", entity_id=source_id)
                source_view: object = source
            else:
                segments = []
                events = []
                source_view = _source_metadata(source)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown source") from exc
        return templates.TemplateResponse(
            request,
            "source.html",
            {
                "session": session,
                "csrf_token": session.csrf_token,
                "source": source_view,
                "segments": segments,
                "events": events,
                "rights_statuses": list(RightsStatus),
            },
        )

    @app.get("/segments/{segment_id}", response_class=HTMLResponse)
    async def segment_page(request: Request, segment_id: str) -> Response:
        session = _html_session(request, signer)
        if session is None:
            return RedirectResponse("/login", status_code=303)
        _require_administrator(session)
        try:
            segment = intake.registry.get_segment(segment_id)
            source = intake.registry.get_source(segment.source_id)
            events = intake.list_audit_events(entity_type="segment", entity_id=segment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown segment") from exc
        return templates.TemplateResponse(
            request,
            "segment.html",
            {
                "session": session,
                "csrf_token": session.csrf_token,
                "segment": segment,
                "source": source,
                "events": events,
            },
        )

    @app.get("/api/v1/review/queue")
    async def api_review_queue(request: Request) -> JSONResponse:
        session = _api_session(request, signer)
        queue = intake.review_queue()
        if not session.can_mutate:
            queue = [item for item in queue if item.entity_type == "source"]
        return _json_response(queue)

    @app.get("/api/v1/sources/{source_id}")
    async def api_source(request: Request, source_id: str) -> JSONResponse:
        session = _api_session(request, signer)
        try:
            source = intake.registry.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown source") from exc
        if not session.can_mutate:
            return _json_response({"source": _source_metadata(source)})
        return _json_response(
            {
                "source": source,
                "audit_events": intake.list_audit_events(entity_type="source", entity_id=source_id),
            }
        )

    @app.post("/api/v1/sources/{source_id}/rights")
    async def api_rights(request: Request, source_id: str) -> Response:
        session, payload = await _authenticated_mutation(request, signer)
        evidence = _evidence(payload)
        try:
            updated = intake.set_rights(
                source_id,
                RightsStatus(_required_text(payload, "status")),
                license_notes=_optional_text(payload, "license_notes"),
                actor=session.actor,
                reason=_required_text(payload, "reason"),
                evidence_reference=evidence,
                expected_revision=_revision(payload),
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _mutation_response(request, updated, f"/sources/{source_id}")

    @app.post("/api/v1/sources/{source_id}/source-decision")
    async def api_source_decision(request: Request, source_id: str) -> Response:
        session, payload = await _authenticated_mutation(request, signer)
        try:
            updated = intake.set_source_approval(
                source_id,
                ApprovalStatus(_decision(payload)),
                actor=session.actor,
                reason=_required_text(payload, "reason"),
                evidence_reference=_evidence(payload),
                expected_revision=_revision(payload),
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _mutation_response(request, updated, f"/sources/{source_id}")

    @app.post("/api/v1/sources/{source_id}/download-decision")
    async def api_download_decision(request: Request, source_id: str) -> Response:
        session, payload = await _authenticated_mutation(request, signer)
        try:
            updated = intake.set_download_approval(
                source_id,
                ApprovalStatus(_decision(payload)),
                actor=session.actor,
                reason=_required_text(payload, "reason"),
                evidence_reference=_evidence(payload),
                expected_revision=_revision(payload),
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _mutation_response(request, updated, f"/sources/{source_id}")

    @app.post("/api/v1/sources/{source_id}/eligibility")
    async def api_eligibility(request: Request, source_id: str) -> Response:
        session, payload = await _authenticated_mutation(request, signer)
        try:
            updated = intake.set_training_eligibility(
                source_id,
                _boolean(payload, "eligible"),
                actor=session.actor,
                reason=_required_text(payload, "reason"),
                evidence_reference=_evidence(payload),
                expected_revision=_revision(payload),
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _mutation_response(request, updated, f"/sources/{source_id}")

    @app.get("/api/v1/sources/{source_id}/segments")
    async def api_segments(request: Request, source_id: str) -> JSONResponse:
        _require_administrator(_api_session(request, signer))
        try:
            segments = intake.list_segments(source_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown source") from exc
        return _json_response(segments)

    @app.post("/api/v1/segments/{segment_id}/decision")
    async def api_segment_decision(request: Request, segment_id: str) -> Response:
        session, payload = await _authenticated_mutation(request, signer)
        try:
            updated = intake.set_segment_approval(
                segment_id,
                ApprovalStatus(_decision(payload)),
                actor=session.actor,
                reason=_required_text(payload, "reason"),
                evidence_reference=_evidence(payload),
                expected_revision=_revision(payload),
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _mutation_response(request, updated, f"/segments/{segment_id}")

    @app.get("/api/v1/media/{entity_id}")
    async def api_media(request: Request, entity_id: str) -> Response:
        _require_administrator(_api_session(request, signer))
        path = resolve_registered_media(intake.registry, entity_id, config.paths.intake_root)
        return media_response(path, request.headers.get("range"))

    @app.post("/api/v1/datasets/build")
    async def api_build_dataset(request: Request) -> Response:
        session, payload = await _authenticated_mutation(request, signer)
        # A build is immutable; the registry itself is its concurrency boundary.
        # expected_revision is still required in the HTTP contract for consistent clients.
        _revision(payload)
        name = _optional_text(payload, "name") or _required_text(payload, "dataset_name")
        try:
            result = intake.build_dataset(
                name,
                actor=session.actor,
                reason=_required_text(payload, "reason"),
                evidence_reference=_evidence(payload),
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _mutation_response(request, result, f"/datasets/{result.name}")

    return app


def _build_service(config: CadenceConfig) -> DatasetIntakeService:
    maximum_bytes = round(config.dataset_intake.unknown_download_reservation_gb * GIB)
    return DatasetIntakeService(
        config,
        downloaders=DownloaderChain(
            [DirectHTTPDownloader(maximum_bytes=maximum_bytes), YtDlpDownloader()]
        ),
        media=FFmpegMediaProcessor(
            config.dataset_intake.ffmpeg_binary, config.dataset_intake.ffprobe_binary
        ),
    )


def _api_session(request: Request, signer: SessionSigner) -> ReviewSession:
    token = request.cookies.get(SESSION_COOKIE)
    if token is None:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        return signer.verify(token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="invalid or expired session") from exc


def _html_session(request: Request, signer: SessionSigner) -> ReviewSession | None:
    try:
        return _api_session(request, signer)
    except HTTPException:
        return None


async def _authenticated_mutation(
    request: Request,
    signer: SessionSigner,
    *,
    require_administrator: bool = True,
) -> tuple[ReviewSession, dict[str, object]]:
    session = _api_session(request, signer)
    if require_administrator:
        _require_administrator(session)
    payload = await _request_payload(request)
    supplied_csrf = request.headers.get("x-csrf-token") or _optional_text(payload, "csrf_token")
    if not verify_csrf(supplied_csrf, session):
        raise HTTPException(status_code=403, detail="invalid CSRF token")
    return session, payload


def _require_administrator(session: ReviewSession) -> None:
    if not session.can_mutate:
        raise HTTPException(
            status_code=403,
            detail="read-only reviewer sessions cannot access this operation",
        )


async def _request_payload(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() == "application/json":
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="request body must be an object")
        return {str(key): value for key, value in payload.items()}
    parsed = parse_qs((await request.body()).decode(), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _required_text(payload: dict[str, object], key: str) -> str:
    value = _optional_text(payload, key)
    if not value:
        raise HTTPException(status_code=422, detail=f"{key} is required")
    return value


def _optional_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _revision(payload: dict[str, object]) -> int:
    value = payload.get("expected_revision")
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail="expected_revision must be an integer")
    if isinstance(value, int):
        revision = value
    elif isinstance(value, str):
        try:
            revision = int(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="expected_revision must be an integer"
            ) from exc
    else:
        raise HTTPException(status_code=422, detail="expected_revision must be an integer")
    if revision < 0:
        raise HTTPException(status_code=422, detail="expected_revision must be non-negative")
    return revision


def _boolean(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(value, str) and value.lower() in {"false", "0", "no", "off"}:
        return False
    raise HTTPException(status_code=422, detail=f"{key} must be a boolean")


def _decision(payload: dict[str, object]) -> str:
    return _optional_text(payload, "decision") or _required_text(payload, "status")


def _evidence(payload: dict[str, object]) -> EvidenceReference | None:
    reference = _optional_text(payload, "evidence_reference")
    if not reference:
        return None
    return EvidenceReference(
        reference=reference,
        description=_optional_text(payload, "evidence_description"),
    )


def _json_response(value: object, *, status_code: int = 200) -> JSONResponse:
    encoded: Any = jsonable_encoder(value)
    return JSONResponse(content=encoded, status_code=status_code)


def _mutation_response(request: Request, value: object, return_path: str) -> Response:
    if _is_json_request(request):
        return _json_response(value)
    return RedirectResponse(return_path, status_code=303)


def _is_json_request(request: Request) -> bool:
    return (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        == "application/json"
    )


def _source_metadata(source: SourceRecord) -> dict[str, object]:
    """Allowlist source fields safe for a read-only reviewer session."""
    return {
        "schema_version": source.schema_version,
        "source_id": source.source_id,
        "revision": source.revision,
        "url": str(source.url),
        "submitted_by": source.submitted_by,
        "collection_method": source.collection_method,
        "submitted_at": source.submitted_at,
        "title": source.title,
        "publisher_or_creator": source.publisher_or_creator,
        "platform": source.platform,
        "duration_seconds": source.duration_seconds,
        "content_length_bytes": source.content_length_bytes,
        "inspection_status": source.inspection_status,
        "download_status": source.download_status,
        "rights_status": source.rights_status,
        "source_approval": source.source_approval,
        "download_approval": source.download_approval,
        "eligible_for_training": source.eligible_for_training,
    }
