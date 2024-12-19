import os
import hashlib
from aider.summary_store import SummaryStore


class FileSummaryStore:
    """Stores file, folder, and architecture summaries using SummaryStore."""

    def __init__(self, root):
        self.file_store = SummaryStore(root, ".aider.file_summaries")
        self.folder_store = SummaryStore(root, ".aider.folder_summaries")
        self.arch_store = SummaryStore(root, ".aider.architecture_summary")

    def reset_file_summaries(self):
        """Reset file summaries to empty."""
        self.file_store.reset_summaries()

    def reset_folder_summaries(self):
        """Reset folder summaries to empty."""
        self.folder_store.reset_summaries()

    def get_all_file_summaries(self):
        """Get all file summaries."""
        return self.file_store.get_all_summaries()

    def get_all_folder_summaries(self):
        """Get all folder summaries."""
        return self.folder_store.get_all_summaries()

    def get_all_architecture_summaries(self):
        """Get all architecture summaries."""
        return self.arch_store.get_all_summaries()

    def reset_architecture_summary(self):
        """Reset architecture summary to empty."""
        self.arch_store.reset_summaries()

    def reset_all_summaries(self):
        """Reset all summaries to empty."""
        self.reset_file_summaries()
        self.reset_folder_summaries()
        self.reset_architecture_summary()

    def get_file_summary(self, rel_fname, content_hash):
        """Get summary for file if content matches."""
        return self.file_store.get_summary(rel_fname, content_hash)

    def save_file_summary(self, rel_fname, content_hash, summary):
        """Save summary for file."""
        if os.path.basename(rel_fname).startswith("."):
            return
        self.file_store.save_summary(rel_fname, content_hash, summary)

    def get_folder_summary(self, folder_path):
        """Get summary for folder if hash matches."""
        return self.folder_store.get_summary(folder_path, None)

    def save_folder_summary(self, folder_path, summary):
        """Save summary for folder."""
        file_hashes = [
            self.file_store.summaries[rel_fname]["hash"]
            for rel_fname in self.file_store.summaries
            if rel_fname.startswith(folder_path)
        ]
        combined_hash = hashlib.sha256("".join(file_hashes).encode()).hexdigest()
        self.folder_store.save_summary(folder_path, combined_hash, summary)

    def get_architecture_summary(self):
        """Get architecture summary if hash matches."""
        return self.arch_store.get_summary("architecture", None)

    def save_architecture_summary(self, summary):
        """Save architecture summary."""
        self.arch_store.save_summary("architecture", "arch_hash", summary)
