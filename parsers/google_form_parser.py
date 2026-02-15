"""Google Form URL → structured JSON parser.

Parses the FB_PUBLIC_LOAD_DATA_ JavaScript variable embedded in Google Forms HTML
to extract form structure, questions, types, options, and validation rules.

Falls back to HTML DOM scraping if the JS variable is unavailable.
"""

import json
import logging
import re
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from models.form_schema import (
    FormQuestion,
    FormSchema,
    FormSection,
    GridConfig,
    QuestionType,
    ScaleConfig,
    ValidationRule,
    GOOGLE_FORM_TYPE_MAP,
)

logger = logging.getLogger(__name__)


def _safe_get(data: Any, *indices: int, default: Any = None) -> Any:
    """Safely traverse nested lists by index without raising."""
    current = data
    for idx in indices:
        if not isinstance(current, list) or idx >= len(current):
            return default
        current = current[idx]
    return current


class GoogleFormParser:
    """Parses a Google Form URL into a structured FormSchema."""

    VIEWFORM_SUFFIX = "/viewform"
    FORMRESPONSE_SUFFIX = "/formResponse"

    def __init__(self, url: str):
        self.original_url = url
        self.url = self._normalize_url(url)
        self.raw_html: Optional[str] = None
        self.raw_data: Optional[list] = None

    def _normalize_url(self, url: str) -> str:
        """Ensure the URL points to the viewform page."""
        url = url.strip()
        # Remove query parameters
        base = url.split("?")[0]
        # Ensure it ends with /viewform
        if base.endswith(self.FORMRESPONSE_SUFFIX):
            base = base.replace(self.FORMRESPONSE_SUFFIX, self.VIEWFORM_SUFFIX)
        if not base.endswith(self.VIEWFORM_SUFFIX):
            base = base.rstrip("/") + self.VIEWFORM_SUFFIX
        return base

    def fetch(self) -> str:
        """Fetch the Google Form HTML."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(self.url, headers=headers, timeout=30)
        response.raise_for_status()
        self.raw_html = response.text
        return self.raw_html

    # ── FB_PUBLIC_LOAD_DATA_ extraction ───────────────────────────────────

    def _extract_fb_public_load_data(self, html: str) -> list:
        """Extract the FB_PUBLIC_LOAD_DATA_ JavaScript variable from the HTML."""
        # Try the standard pattern first
        pattern = r"var\s+FB_PUBLIC_LOAD_DATA_\s*=\s*(.*?);\s*</script>"
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            # Fallback: some forms use a slightly different format
            pattern2 = r"FB_PUBLIC_LOAD_DATA_\s*=\s*(.*?);\s*</script>"
            match = re.search(pattern2, html, re.DOTALL)
        if not match:
            raise ValueError(
                "Could not find FB_PUBLIC_LOAD_DATA_ in the form HTML. "
                "The form may be private, deleted, or the URL may be invalid."
            )
        raw_json = match.group(1).strip()
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse form data JSON: {e}")
        self.raw_data = data
        return data

    # ── Low-level field parsers ───────────────────────────────────────────

    def _parse_question_type(self, type_id: int) -> QuestionType:
        """Map Google's internal type ID to our QuestionType enum."""
        return GOOGLE_FORM_TYPE_MAP.get(type_id, QuestionType.UNSUPPORTED)

    def _parse_options(self, choice_data: list) -> tuple[list[str], bool]:
        """Parse options from a question's choice data. Returns (options, has_other)."""
        options: list[str] = []
        has_other = False
        if not choice_data:
            return options, has_other
        for choice in choice_data:
            if not isinstance(choice, list) or len(choice) == 0:
                continue
            option_text = choice[0]
            if option_text is not None:
                options.append(str(option_text))
            # "Other" flag can appear at index 4 (value 1) or as a
            # special sentinel where the text is None and a later
            # element marks it.
            if len(choice) > 4 and choice[4] == 1:
                has_other = True
        return options, has_other

    def _parse_scale_config(self, question_data: list) -> Optional[ScaleConfig]:
        """Parse linear scale configuration from a question sub-array."""
        try:
            choices = _safe_get(question_data, 1, default=[])
            if not isinstance(choices, list) or not choices:
                return ScaleConfig()

            # choices is a list of [value_str, ...] sub-lists
            values: list[int] = []
            for opt in choices:
                if isinstance(opt, list) and len(opt) > 0:
                    try:
                        values.append(int(opt[0]))
                    except (ValueError, TypeError):
                        pass
                # Sometimes options are bare strings like "1", "2", …
                elif isinstance(opt, (int, float)):
                    values.append(int(opt))
                elif isinstance(opt, str):
                    try:
                        values.append(int(opt))
                    except ValueError:
                        pass

            if not values:
                return ScaleConfig()

            min_val = min(values)
            max_val = max(values)

            # Scale labels live in a sibling array; exact position varies
            min_label = None
            max_label = None
            # Try question_data[3] — some forms store labels there
            labels_candidate = _safe_get(question_data, 3, default=None)
            if isinstance(labels_candidate, list) and len(labels_candidate) >= 2:
                if labels_candidate[0]:
                    min_label = str(labels_candidate[0])
                if labels_candidate[1]:
                    max_label = str(labels_candidate[1])

            return ScaleConfig(
                min_value=min_val,
                max_value=max_val,
                min_label=min_label,
                max_label=max_label,
            )
        except (IndexError, TypeError):
            return ScaleConfig()

    def _parse_grid_config(self, question_groups: list) -> Optional[GridConfig]:
        """Parse grid question configuration (rows and columns).

        For grid questions, each sub-question in *question_groups* represents
        one row.  The columns are the options shared across all rows.
        """
        try:
            rows: list[str] = []
            columns: list[str] = []

            for sub_q in question_groups:
                if not isinstance(sub_q, list) or len(sub_q) < 2:
                    continue

                # Row label: the sub-question's own "title" is typically
                # stored as a plain string at sub_q[0] or inside a nested
                # list at sub_q[0][0].
                row_label = sub_q[0]
                if isinstance(row_label, list) and len(row_label) > 0:
                    row_label = row_label[0]
                if row_label is not None:
                    rows.append(str(row_label))

                # Columns from the first sub-question's options list
                if not columns:
                    col_data = _safe_get(sub_q, 1, default=[])
                    if isinstance(col_data, list):
                        for col in col_data:
                            if isinstance(col, list) and len(col) > 0 and col[0] is not None:
                                columns.append(str(col[0]))

            return GridConfig(rows=rows, columns=columns) if rows or columns else None
        except (IndexError, TypeError):
            return None

    def _parse_validation(self, constraint_data: Any) -> Optional[ValidationRule]:
        """Parse validation / constraint rules for a question."""
        if not constraint_data or not isinstance(constraint_data, list):
            return None
        try:
            # Constraints are typically nested one level deep
            rule_list = constraint_data
            if isinstance(constraint_data[0], list):
                rule_list = constraint_data[0]

            rule_type = str(rule_list[0]) if len(rule_list) > 0 and rule_list[0] is not None else None
            condition = str(rule_list[1]) if len(rule_list) > 1 and rule_list[1] is not None else None
            value = rule_list[2] if len(rule_list) > 2 else None
            error_msg = rule_list[3] if len(rule_list) > 3 else None
            return ValidationRule(
                rule_type=rule_type,
                condition=condition,
                value=value,
                error_message=str(error_msg) if error_msg else None,
            )
        except (IndexError, TypeError):
            return None

    # ── Main parse logic ──────────────────────────────────────────────────

    def parse(self) -> FormSchema:
        """Parse the Google Form and return a structured FormSchema.

        Tries the FB_PUBLIC_LOAD_DATA_ JS variable first.  If that yields
        zero questions, falls back to HTML DOM scraping.
        """
        if not self.raw_html:
            self.fetch()

        try:
            schema = self._parse_from_fb_data()
            if schema.questions:
                return schema
            logger.warning(
                "FB_PUBLIC_LOAD_DATA_ parsed but yielded 0 questions; "
                "falling back to HTML scraping."
            )
        except ValueError:
            logger.warning(
                "FB_PUBLIC_LOAD_DATA_ not found; falling back to HTML scraping."
            )

        return self._parse_from_html()

    # ── Strategy 1: FB_PUBLIC_LOAD_DATA_ ──────────────────────────────────

    def _parse_from_fb_data(self) -> FormSchema:
        """Parse form structure from the FB_PUBLIC_LOAD_DATA_ JS variable."""
        data = self._extract_fb_public_load_data(self.raw_html)

        # ── Form metadata ─────────────────────────────────────────────
        form_info = _safe_get(data, 1, default=[])
        form_title = _safe_get(form_info, 8, default="Untitled Form")
        if not form_title:
            form_title = "Untitled Form"
        form_title = str(form_title)

        form_description = None
        desc_candidate = _safe_get(form_info, 0, default=None)
        if desc_candidate:
            form_description = str(desc_candidate)

        # ── Questions ─────────────────────────────────────────────────
        questions_data = _safe_get(form_info, 1, default=[])
        questions: list[FormQuestion] = []
        sections: list[FormSection] = []
        current_section = 0

        if not questions_data or not isinstance(questions_data, list):
            return FormSchema(
                form_url=self.original_url,
                form_title=form_title,
                form_description=form_description,
                questions=[],
                sections=[],
            )

        for idx, item in enumerate(questions_data):
            if not isinstance(item, list):
                continue

            item_title = _safe_get(item, 1, default=None)
            item_desc = _safe_get(item, 2, default=None)

            # item[4] holds the list of sub-question groups (one per
            # question, or multiple for grids).
            question_groups = _safe_get(item, 4, default=None)

            if not question_groups or not isinstance(question_groups, list):
                # Section break / description-only block
                if item_title:
                    sections.append(
                        FormSection(
                            section_index=current_section,
                            title=str(item_title),
                            description=str(item_desc) if item_desc else None,
                        )
                    )
                    current_section += 1
                continue

            # Detect grid questions: multiple sub-groups that share the
            # same question type (7 or 8) indicate a grid whose rows are
            # the sub-groups.
            first_type = _safe_get(question_groups, 0, 3, default=None)
            is_grid = (
                first_type in (7, 8)
                or (len(question_groups) > 1
                    and all(
                        _safe_get(qg, 3, default=None) == first_type
                        for qg in question_groups
                        if isinstance(qg, list)
                    )
                    and first_type in (2, 4, 7, 8))
            )

            if is_grid and first_type in (7, 8):
                # Build a single grid question from all sub-groups
                grid_config = self._parse_grid_config(question_groups)
                q_type = self._parse_question_type(first_type)
                first_qg = question_groups[0]
                entry_id = str(_safe_get(first_qg, 0, default=f"q_{idx}"))
                is_required = self._extract_required(first_qg)

                questions.append(
                    FormQuestion(
                        question_id=f"q_{entry_id}",
                        entry_id=f"entry.{entry_id}",
                        question_text=str(item_title) if item_title else f"Question {idx + 1}",
                        question_type=q_type,
                        options=[],
                        is_required=is_required,
                        grid_config=grid_config,
                        section_index=current_section,
                    )
                )
                continue

            # Normal (non-grid) questions — usually one sub-group per item
            for q_group in question_groups:
                if not isinstance(q_group, list) or len(q_group) < 1:
                    continue

                entry_id = str(_safe_get(q_group, 0, default=f"q_{idx}"))
                type_id = _safe_get(q_group, 3, default=0)
                if not isinstance(type_id, int):
                    try:
                        type_id = int(type_id)
                    except (ValueError, TypeError):
                        type_id = 0
                question_type = self._parse_question_type(type_id)

                is_required = self._extract_required(q_group)

                # Options
                options: list[str] = []
                has_other = False
                raw_choices = _safe_get(q_group, 1, default=None)
                if isinstance(raw_choices, list):
                    options, has_other = self._parse_options(raw_choices)

                # Scale config
                scale_config = None
                if question_type == QuestionType.LINEAR_SCALE:
                    scale_config = self._parse_scale_config(q_group)

                # Grid config (shouldn't normally reach here for grids,
                # but handle defensively)
                grid_config = None
                if question_type in (
                    QuestionType.MULTIPLE_CHOICE_GRID,
                    QuestionType.CHECKBOX_GRID,
                ):
                    grid_config = self._parse_grid_config(question_groups)

                # Validation
                validation = None
                constraint_data = _safe_get(q_group, 4, default=None)
                if isinstance(constraint_data, list):
                    validation = self._parse_validation(constraint_data)

                questions.append(
                    FormQuestion(
                        question_id=f"q_{entry_id}",
                        entry_id=f"entry.{entry_id}",
                        question_text=str(item_title) if item_title else f"Question {idx + 1}",
                        question_type=question_type,
                        options=options,
                        is_required=is_required,
                        validation=validation,
                        scale_config=scale_config,
                        grid_config=grid_config,
                        has_other_option=has_other,
                        section_index=current_section,
                    )
                )

        if not sections:
            sections.append(
                FormSection(
                    section_index=0,
                    title=form_title,
                    description=form_description,
                )
            )

        return FormSchema(
            form_url=self.original_url,
            form_title=form_title,
            form_description=form_description,
            questions=questions,
            sections=sections,
        )

    @staticmethod
    def _extract_required(q_group: list) -> bool:
        """Extract the required flag from a question sub-group."""
        constraint = _safe_get(q_group, 4, default=None)
        if constraint is None:
            return False
        if isinstance(constraint, list):
            # The required flag is typically at index 2 inside the
            # constraint sub-array.
            flag = _safe_get(constraint, 2, default=0)
            return bool(flag)
        return bool(constraint)

    # ── Strategy 2: HTML DOM fallback ─────────────────────────────────────

    def _parse_from_html(self) -> FormSchema:
        """Fallback parser that scrapes the rendered HTML DOM.

        This is less reliable than the JS variable but works for forms
        where the variable is stripped or obfuscated.
        """
        soup = BeautifulSoup(self.raw_html, "html.parser")

        # Title
        title_el = soup.find("meta", property="og:title")
        form_title = title_el["content"] if title_el and title_el.get("content") else "Untitled Form"

        # Description
        desc_el = soup.find("meta", property="og:description")
        form_description = desc_el["content"] if desc_el and desc_el.get("content") else None

        questions: list[FormQuestion] = []
        question_idx = 0

        # Google Forms renders each question inside a div with
        # data-params containing a JSON-like structure.
        for div in soup.find_all("div", attrs={"data-params": True}):
            try:
                raw = div["data-params"]
                # data-params starts with "%." prefix sometimes
                if raw.startswith("%."):
                    raw = raw[2:]
                # It's a JS array literal; wrap in brackets if needed
                if not raw.startswith("["):
                    raw = "[" + raw + "]"
                params = json.loads(raw)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            # params structure: [null, question_text, desc, entry_id, ...]
            q_text = _safe_get(params, 1, default=None)
            if not q_text:
                # Try nested
                q_text = _safe_get(params, 0, 1, default=f"Question {question_idx + 1}")

            entry_id = _safe_get(params, 4, 0, 0, default=str(question_idx))

            questions.append(
                FormQuestion(
                    question_id=f"q_{entry_id}",
                    entry_id=f"entry.{entry_id}",
                    question_text=str(q_text),
                    question_type=QuestionType.SHORT_TEXT,
                    is_required=False,
                    section_index=0,
                )
            )
            question_idx += 1

        # If data-params didn't work, try finding input/textarea elements
        if not questions:
            for inp in soup.find_all(["input", "textarea", "select"]):
                name = inp.get("name", "")
                if not name.startswith("entry."):
                    continue
                entry_id = name.replace("entry.", "")
                # Try to find the label
                label = ""
                parent = inp.find_parent("div", class_=re.compile(r"freebirdFormviewItem"))
                if parent:
                    label_el = parent.find(
                        "span", class_=re.compile(r"freebirdFormviewItem.*QuestionText")
                    )
                    if label_el:
                        label = label_el.get_text(strip=True)
                if not label:
                    label = f"Question (entry.{entry_id})"

                q_type = QuestionType.SHORT_TEXT
                tag = inp.name
                if tag == "textarea":
                    q_type = QuestionType.PARAGRAPH
                elif tag == "select":
                    q_type = QuestionType.DROPDOWN

                options: list[str] = []
                if tag == "select":
                    for opt in inp.find_all("option"):
                        val = opt.get_text(strip=True)
                        if val:
                            options.append(val)

                questions.append(
                    FormQuestion(
                        question_id=f"q_{entry_id}",
                        entry_id=f"entry.{entry_id}",
                        question_text=label,
                        question_type=q_type,
                        options=options,
                        is_required=inp.get("required") is not None,
                        section_index=0,
                    )
                )

        sections = [
            FormSection(
                section_index=0,
                title=form_title,
                description=form_description,
            )
        ]

        return FormSchema(
            form_url=self.original_url,
            form_title=form_title,
            form_description=form_description,
            questions=questions,
            sections=sections,
        )
