import json
from pathlib import Path


class SummaryStore:
    """Handles loading, saving, and managing summaries for a given store file."""

    def __init__(self, root, filename):
        self.file = Path(root) / filename
        self.summaries = self._load_summaries()

    def reset_summaries(self):
        """Reset summaries to empty and save."""
        self.summaries = {}
        self._save_summaries()

    def _load_summaries(self):
        """Load summaries from file."""
        if not self.file.exists():
            return {}
        try:
            with open(self.file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading summaries from {self.file}: {e}")
            return {}

    def _save_summaries(self):
        """Save summaries to file with error handling."""
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.summaries, f, indent=2, sort_keys=True)
            temp_file.replace(self.file)
        except Exception as e:
            print(f"Error saving summaries to {self.file}: {e}")

    def get_all_summaries(self):
        """Return all summaries as a dictionary."""
        return self.summaries.copy()

    def get_summary(self, key, content_hash):
        """Get summary for a key if content matches."""
        if key not in self.summaries:
            return None
        entry = self.summaries[key]
        if entry["hash"] != content_hash:
            return None
        return entry["summary"]

    def save_summary(self, key, content_hash, summary):
        """Save summary for a key."""
        self.summaries[key] = {"hash": content_hash, "summary": summary}
        self._save_summaries()

    def clean_duplicates(self):
        """Remove any duplicate entries by keeping the most recent for each key."""
        cleaned = {}
        for key, entries in self.summaries.items():
            if isinstance(entries, dict):
                cleaned[key] = entries
            elif isinstance(entries, list) and entries:
                cleaned[key] = entries[-1]
        if cleaned != self.summaries:
            self.summaries = cleaned
            self._save_summaries()
