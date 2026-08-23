"""CSV export for survey datasets."""

from typing import Optional

import pandas as pd

from models.form_schema import FormSchema
from models.response import SurveyDataset


class CSVExporter:
    """Export survey dataset to CSV format."""

    @staticmethod
    def to_dataframe(
        dataset: SurveyDataset,
        form_schema: FormSchema,
        existing_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Convert a SurveyDataset to a pandas DataFrame.

        Args:
            dataset: The generated survey dataset.
            form_schema: The form schema for column naming.
            existing_data: Optional existing real data to append to.

        Returns:
            A pandas DataFrame with all responses.
        """
        # Build question ID to text mapping
        q_map = {q.question_id: q.question_text for q in form_schema.questions}

        responses = dataset.responses
        has_waves = any(getattr(r, "wave", 1) != 1 for r in responses)
        has_stimulus = any(getattr(r, "stimulus_variant", None) for r in responses)

        rows = []
        for resp in responses:
            row = {}
            for qid, answer in resp.answers.items():
                col_name = q_map.get(qid, qid)
                if isinstance(answer, list):
                    row[col_name] = "; ".join(str(a) for a in answer)
                elif isinstance(answer, dict):
                    # Grid answers: flatten to "row: value" pairs
                    for grid_row, grid_val in answer.items():
                        if isinstance(grid_val, list):
                            row[f"{col_name} [{grid_row}]"] = "; ".join(
                                str(v) for v in grid_val
                            )
                        else:
                            row[f"{col_name} [{grid_row}]"] = str(grid_val)
                else:
                    row[col_name] = answer

            row["is_synthetic"] = True
            row["persona_id"] = resp.persona_id
            row["persona_summary"] = resp.persona_summary
            row["generation_timestamp"] = resp.generation_timestamp
            # Phase 2 research metadata (columns present only when in use).
            if has_waves:
                row["wave"] = getattr(resp, "wave", 1)
            if has_stimulus:
                row["stimulus_variant"] = getattr(resp, "stimulus_variant", None)
            for tname, tval in getattr(resp, "latent_traits", {}).items():
                row[f"trait_{tname}"] = tval

            rows.append(row)

        synthetic_df = pd.DataFrame(rows)

        if existing_data is not None:
            # Mark existing data as not synthetic
            existing_data = existing_data.copy()
            if "is_synthetic" not in existing_data.columns:
                existing_data["is_synthetic"] = False
            if "persona_id" not in existing_data.columns:
                existing_data["persona_id"] = ""
            if "persona_summary" not in existing_data.columns:
                existing_data["persona_summary"] = ""
            if "generation_timestamp" not in existing_data.columns:
                existing_data["generation_timestamp"] = ""

            combined = pd.concat([existing_data, synthetic_df], ignore_index=True)
            return combined

        return synthetic_df

    @staticmethod
    def to_csv_string(df: pd.DataFrame) -> str:
        """Convert DataFrame to CSV string."""
        return df.to_csv(index=False)

    @staticmethod
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to CSV bytes for download."""
        return df.to_csv(index=False).encode("utf-8")
