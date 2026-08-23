"""Framework-agnostic survey-response generation.

The Streamlit UI is a thin adapter over this service: it drives ``run()`` and
renders the ``GenerationProgress`` events it yields. Keeping the orchestration
here (rather than inside the page) makes it unit-testable and lets long runs be
checkpointed to disk so they can be resumed after an interruption.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from config import Settings
from generators.persona_generator import PersonaGenerator
from generators.response_generator import ResponseGenerator
from models.form_schema import FormSchema
from models.persona import Persona
from models.response import GeneratedResponse, SurveyDataset
from services.traits import sample_traits
from utils.llm_client import LLMClient, RateLimiter, classify_error

log = logging.getLogger(__name__)

# Where partial runs are persisted so they survive an interruption. Gitignored.
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / ".synthsurvey_checkpoints"

# Stop early if this many responses in a row fail while nothing has succeeded
# (usually a bad key or a dead model).
_MAX_CONSECUTIVE_FAIL = 10


@dataclass
class GenerationProgress:
    """One completed unit of work, emitted by ``GenerationService.run``."""

    completed: int
    total: int
    generated: int
    failed: int
    persona: Optional[Persona]
    response: Optional[GeneratedResponse]
    entry: dict


def run_key(form_url: str, provider: str, model: str, constraints: Optional[str]) -> str:
    """Stable id for a run configuration, used to name its checkpoint file."""
    raw = "|".join([form_url or "", provider or "", model or "", constraints or ""])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def peek_checkpoint(
    schema: FormSchema,
    settings: Settings,
    constraints: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> int:
    """Return how many responses are already saved for this run config (0 if none).

    Cheap — reads only the checkpoint file, without constructing an LLM client.
    """
    provider = settings.llm_provider
    model = getattr(settings, f"{provider}_model", "unknown")
    key = run_key(schema.form_url, provider, model, constraints or None)
    path = (Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR) / f"{key}.jsonl"
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


class GenerationService:
    """Generates survey responses concurrently, with optional checkpointing."""

    def __init__(
        self,
        schema: FormSchema,
        settings: Settings,
        constraints: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        checkpoint_dir: Optional[Path] = None,
        resume: bool = False,
        quota_plan: Optional[list[dict]] = None,
        stimulus_plan: Optional[list[dict]] = None,
        waves: int = 1,
        enable_traits: bool = False,
        traits_seed: int = 42,
    ):
        self.schema = schema
        self.settings = settings
        self.constraints = constraints or None
        self.quota_plan = quota_plan or []
        self.stimulus_plan = stimulus_plan or []
        self.waves = max(1, int(waves))
        self.enable_traits = enable_traits
        self.traits_seed = traits_seed
        self.provider = settings.llm_provider
        self.model_name = getattr(settings, f"{self.provider}_model", "unknown")

        rate = (1.0 / settings.api_call_delay) if settings.api_call_delay and settings.api_call_delay > 0 else 0.0
        self.rate_limiter = RateLimiter(rate)
        # Building the client validates the API key and may raise ValueError.
        self.llm = llm_client or LLMClient(settings, rate_limiter=self.rate_limiter)
        self.persona_gen = PersonaGenerator(self.llm, settings)
        self.response_gen = ResponseGenerator(self.llm, settings)

        self.dataset = SurveyDataset(
            form_title=schema.form_title,
            form_url=schema.form_url,
            generation_started=datetime.now(timezone.utc).isoformat(),
        )
        self.report_log: list[dict] = []
        self.stopped_reason: Optional[str] = None  # None | "user" | "failures"

        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
        self._key = run_key(schema.form_url, self.provider, self.model_name, self.constraints)
        self.checkpoint_path = self._checkpoint_dir / f"{self._key}.jsonl"

        if resume:
            self._load_checkpoint()
        else:
            # Fresh run — discard any stale partial for this exact config.
            self.clear_checkpoint()

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        for line in self.checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.dataset.add_response(GeneratedResponse(**json.loads(line)))
            except Exception:
                log.warning("Skipping unreadable checkpoint line", exc_info=True)

    def _append_checkpoint(self, response: GeneratedResponse) -> None:
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(response.model_dump()) + "\n")
        except OSError:
            log.warning("Failed to write checkpoint", exc_info=True)

    def clear_checkpoint(self) -> None:
        try:
            self.checkpoint_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("Failed to clear checkpoint", exc_info=True)

    def completed_count(self) -> int:
        return len(self.dataset.responses)

    # ── Work ──────────────────────────────────────────────────────────────

    def _stimulus_for(self, i: int) -> Optional[dict]:
        return self.stimulus_plan[i] if i < len(self.stimulus_plan) else None

    def _blank_entry(self, i: int) -> dict:
        return {
            "response_num": i + 1,
            "status": "success",
            "persona_name": None,
            "error_type": None,
            "error_message": None,
            "suggestion": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fail_entry(self, entry: dict, exc: Exception) -> dict:
        diag = classify_error(exc, self.provider, self.model_name)
        entry["status"] = "failed"
        entry["error_type"] = diag["error_type"]
        entry["error_message"] = diag["message"][:300]
        entry["suggestion"] = diag["suggestion"]
        return entry

    def _work(self, i: int) -> dict:
        """Wave 1: generate persona i (traits/quota) + its first response."""
        entry = self._blank_entry(i)
        try:
            assignment = self.quota_plan[i] if i < len(self.quota_plan) else None
            traits = sample_traits(i, self.traits_seed) if self.enable_traits else None
            persona = self.persona_gen.generate_one(
                self.schema, self.constraints, assignment=assignment, latent_traits=traits
            )
            entry["persona_name"] = persona.name
            response = self.response_gen.generate_one(
                persona, self.schema, stimulus=self._stimulus_for(i), wave=1
            )
            if not response.generation_success:
                entry["status"] = "failed"
                entry["error_type"] = "Validation Failed"
                entry["error_message"] = "Response generated but failed validation checks"
                entry["suggestion"] = "Try lowering temperature or using a more capable model"
            return {"i": i, "entry": entry, "persona": persona, "response": response}
        except Exception as exc:  # noqa: BLE001 - classify and report, never crash the run
            return {"i": i, "entry": self._fail_entry(entry, exc),
                    "persona": None, "response": None}

    def _respond_work(self, persona: Persona, i: int, wave: int) -> dict:
        """Later waves: reuse persona i to answer again with realistic drift."""
        entry = self._blank_entry(i)
        entry["persona_name"] = persona.name
        try:
            response = self.response_gen.generate_one(
                persona, self.schema, stimulus=self._stimulus_for(i), wave=wave
            )
            if not response.generation_success:
                entry["status"] = "failed"
                entry["error_type"] = "Validation Failed"
                entry["error_message"] = "Response generated but failed validation checks"
                entry["suggestion"] = "Try lowering temperature or using a more capable model"
            return {"i": i, "entry": entry, "persona": persona, "response": response}
        except Exception as exc:  # noqa: BLE001
            return {"i": i, "entry": self._fail_entry(entry, exc),
                    "persona": persona, "response": None}

    def run(
        self,
        count: int,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[GenerationProgress]:
        """Generate responses, yielding one progress event each.

        ``count`` is the number of *personas*. With ``waves == 1`` that equals
        the number of responses (and resume tops up toward it). With
        ``waves > 1`` each persona answers once per wave, for ``count * waves``
        responses total; multi-wave runs are always fresh (resume is single-wave).
        """
        concurrency = max(1, int(getattr(self.settings, "concurrency", 4)))
        total = count * self.waves

        state = {
            "completed": len(self.dataset.responses),
            "generated": sum(1 for r in self.dataset.responses if r.generation_success),
            "consecutive": 0,
        }
        state["failed"] = state["completed"] - state["generated"]

        def record(result: dict, wave: int) -> GenerationProgress:
            entry = result["entry"]
            entry["wave"] = wave
            entry["response_num"] = (wave - 1) * count + result["i"] + 1
            response = result["response"]
            state["completed"] += 1
            if response is not None:
                self.dataset.add_response(response)
                self._append_checkpoint(response)
                if response.generation_success:
                    state["generated"] += 1
                    state["consecutive"] = 0
                else:
                    state["failed"] += 1
                    state["consecutive"] += 1
            else:
                state["failed"] += 1
                state["consecutive"] += 1
            self.report_log.append(entry)
            return GenerationProgress(
                completed=state["completed"], total=total,
                generated=state["generated"], failed=state["failed"],
                persona=result["persona"], response=response, entry=entry,
            )

        # ── Wave 1: personas + first responses ────────────────────────────
        personas: dict[int, Persona] = {}
        start = state["completed"] if self.waves == 1 else 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(self._work, i): i for i in range(start, count)}
            for future in as_completed(futures):
                if should_stop is not None and should_stop():
                    for f in futures:
                        f.cancel()
                    self.stopped_reason = "user"
                    break
                result = future.result()
                if result["persona"] is not None:
                    personas[result["i"]] = result["persona"]
                yield record(result, wave=1)
                if state["consecutive"] >= _MAX_CONSECUTIVE_FAIL and state["generated"] == 0:
                    for f in futures:
                        f.cancel()
                    self.stopped_reason = "failures"
                    break

        # ── Later waves: reuse personas with drift ────────────────────────
        for wave in range(2, self.waves + 1):
            if self.stopped_reason is not None:
                break
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(self._respond_work, personas[i], i, wave): i
                    for i in sorted(personas)
                }
                for future in as_completed(futures):
                    if should_stop is not None and should_stop():
                        for f in futures:
                            f.cancel()
                        self.stopped_reason = "user"
                        break
                    yield record(future.result(), wave=wave)

        self.dataset.generation_completed = datetime.now(timezone.utc).isoformat()
        self.dataset.total_failed = state["failed"]
        # Results arrive out of order under concurrency — restore submission order.
        self.report_log.sort(key=lambda e: e["response_num"])

        # A single-wave run that finished cleanly no longer needs its checkpoint.
        if self.stopped_reason is None and self.waves == 1 and state["completed"] >= count:
            self.clear_checkpoint()
