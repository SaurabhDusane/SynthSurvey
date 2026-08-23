"""Optional Google Sheets export for survey datasets."""


import pandas as pd

from config import Settings


class SheetsExporter:
    """Export survey data to Google Sheets (optional feature)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _init_client(self):
        """Initialize the gspread client."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            creds_file = self.settings.google_sheets_credentials_file
            if not creds_file:
                raise ValueError(
                    "GOOGLE_SHEETS_CREDENTIALS_FILE not set in environment."
                )

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            credentials = Credentials.from_service_account_file(creds_file, scopes=scopes)
            self._client = gspread.authorize(credentials)
        except ImportError:
            raise ImportError(
                "gspread and google-auth are required for Google Sheets export. "
                "Install them with: pip install gspread google-auth"
            )

    def export(
        self,
        df: pd.DataFrame,
        spreadsheet_name: str,
        worksheet_name: str = "Synthetic Responses",
    ) -> str:
        """Export a DataFrame to a Google Sheet.

        Args:
            df: The DataFrame to export.
            spreadsheet_name: Name of the Google Spreadsheet to create/update.
            worksheet_name: Name of the worksheet tab.

        Returns:
            URL of the created/updated spreadsheet.
        """
        if not self._client:
            self._init_client()

        try:
            spreadsheet = self._client.open(spreadsheet_name)
        except Exception:
            spreadsheet = self._client.create(spreadsheet_name)

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            worksheet.clear()
        except Exception:
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name,
                rows=len(df) + 1,
                cols=len(df.columns),
            )

        # Convert DataFrame to list of lists
        data = [df.columns.tolist()] + df.fillna("").values.tolist()
        worksheet.update(range_name="A1", values=data)

        return spreadsheet.url
