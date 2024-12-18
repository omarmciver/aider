import colorsys
import hashlib
import json
import math
import mimetypes
import os
import random
import shutil
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

from aider.dump import dump
from aider.special import filter_important_files
from aider.utils import Spinner

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


class FileSummaryStore:
    """Stores file summaries persistently with deduplication"""

    def __init__(self, root):
        self.root = root
        self.file = Path(root) / ".aider.file_summaries"
        self.folder_file = Path(root) / ".aider.folder_summaries"
        self.architecture_file = Path(root) / ".aider.architecture_summary"

        self.summaries = self._load_summaries()
        self.folder_summaries, self.folder_hashes = self._load_folder_summaries()
        self.architecture_summary = self._load_architecture_summary()

        self._clean_duplicates()

    def _load_folder_summaries(self):
        """Load folder summaries from file"""
        if not self.folder_file.exists():
            return {}, {}
        try:
            with open(self.folder_file) as f:
                data = json.load(f)
                return data.get("summaries", {}), data.get("hashes", {})
        except Exception as e:
            if self.verbose:
                print(f"Error loading folder summaries: {e}")
            return {}

    def _load_architecture_summary(self):
        """Load architecture summary from file"""
        if not self.architecture_file.exists():
            return None
        try:
            with open(self.architecture_file) as f:
                return f.read()
        except Exception as e:
            if self.verbose:
                print(f"Error loading architecture summary: {e}")
            return None

    def _save_folder_summaries(self):
        """Save folder summaries to file with error handling"""
        try:
            self.folder_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.folder_file, "w") as f:
                json.dump(
                    {"summaries": self.folder_summaries, "hashes": self.folder_hashes},
                    f,
                    indent=2,
                    sort_keys=True,
                )
        except Exception as e:
            if self.verbose:
                print(f"Error saving folder summaries: {e}")

    def _save_architecture_summary(self):
        """Save architecture summary to file with error handling"""
        try:
            self.architecture_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.architecture_file, "w") as f:
                f.write(self.architecture_summary)
        except Exception as e:
            if self.verbose:
                print(f"Error saving architecture summary: {e}")

    def get_folder_summary(self, folder_path):
        """Get summary for folder if exists"""
        return self.folder_summaries.get(folder_path)

    def save_folder_summary(self, folder_path, summary):
        """Save summary for folder"""
        # Calculate combined hash of file hashes in the folder
        file_hashes = [
            self.summaries[rel_fname]["hash"]
            for rel_fname in self.summaries
            if rel_fname.startswith(folder_path)
        ]
        combined_hash = hashlib.sha256("".join(file_hashes).encode()).hexdigest()

        # Calculate combined hash of file hashes in the folder
        file_hashes = [
            self.summaries[rel_fname]["hash"]
            for rel_fname in self.summaries
            if rel_fname.startswith(folder_path)
        ]
        combined_hash = hashlib.sha256("".join(file_hashes).encode()).hexdigest()

        self.folder_summaries[folder_path] = summary
        self.folder_hashes[folder_path] = combined_hash
        self._save_folder_summaries()
        self.folder_hashes[folder_path] = combined_hash
        self._save_folder_summaries()

    def get_architecture_summary(self):
        """Get architecture summary if exists"""
        return self.architecture_summary

    def save_architecture_summary(self, summary):
        """Save architecture summary"""
        self.architecture_summary = summary
        self._save_architecture_summary()

    def _load_summaries(self):
        """Load summaries from file"""
        if not self.file.exists():
            return {}
        try:
            with open(self.file) as f:
                return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"Error loading file summaries: {e}")
            return {}

    def _clean_duplicates(self):
        """Remove any duplicate entries by keeping most recent for each file"""
        cleaned = {}
        for rel_fname, entries in self.summaries.items():
            if isinstance(entries, dict):
                # Single entry format
                cleaned[rel_fname] = entries
            elif isinstance(entries, list):
                # Keep only the last entry in the list
                if entries:
                    cleaned[rel_fname] = entries[-1]

        if cleaned != self.summaries:
            self.summaries = cleaned
            self._save_summaries()

    def _save_summaries(self):
        """Save summaries to file with error handling"""
        try:
            # Create parent directories if they don't exist
            self.file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_file = self.file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.summaries, f, indent=2, sort_keys=True)

            # Rename temporary file to actual file
            temp_file.replace(self.file)
        except Exception as e:
            if self.verbose:
                print(f"Error saving file summaries: {e}")

    def get_summary(self, rel_fname, content_hash):
        """Get summary for file if content matches"""
        if rel_fname not in self.summaries:
            return None
        entry = self.summaries[rel_fname]
        if entry["hash"] != content_hash:
            return None
        return entry["summary"]

    def save_summary(self, rel_fname, content_hash, summary):
        """Save summary for file with timestamp"""
        # Skip saving summaries for hidden files
        if os.path.basename(rel_fname).startswith("."):
            return

        self.summaries[rel_fname] = {"hash": content_hash, "summary": summary}
        self._save_summaries()


SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh
        self.file_summaries = None

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None

        if self.verbose:
            self.io.tool_output(
                f"RepoMap initialized with map_mul_no_files: {self.map_mul_no_files}"
            )

    def token_count(self, text):
        len_text = len(text)
        if len_text < 200:
            return self.main_model.token_count(text)

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        step = num_lines // 100 or 1
        lines = lines[::step]
        sample_text = "".join(lines)
        sample_tokens = self.main_model.token_count(sample_text)
        est_tokens = sample_tokens / len(sample_text) * len_text
        return est_tokens

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            # Issue #1288: ValueError: path is on mount 'C:', start on mount 'D:'
            # Just return the full fname.
            return fname

    def tags_cache_error(self, original_error=None):
        """Handle SQLite errors by trying to recreate cache, falling back to dict if needed"""

        if self.verbose and original_error:
            self.io.tool_warning(f"Tags cache error: {str(original_error)}")

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return

        path = Path(self.root) / self.TAGS_CACHE_DIR

        # Try to recreate the cache
        try:
            # Delete existing cache dir
            if path.exists():
                shutil.rmtree(path)

            # Try to create new cache
            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            # If we got here, the new cache works
            self.TAGS_CACHE = new_cache
            return

        except SQLITE_ERRORS as e:
            # If anything goes wrong, warn and fall back to dict
            self.io.tool_warning(
                f"Unable to use tags cache at {path}, falling back to memory cache"
            )
            if self.verbose:
                self.io.tool_warning(f"Cache recreation error: {str(e)}")

        self.TAGS_CACHE = dict()

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_warning(f"File not found error: {fname}")

    def reload_file_summaries(self):
        self.file_summaries = None

    def reload_folder_summaries(self):
        """Reload folder summaries from file"""
        self.folder_summaries = self._load_folder_summaries()

    def reload_arch_summary(self):
        """Reload architecture summary from file"""
        self.architecture_summary = self._load_architecture_summary()

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        try:
            val = self.TAGS_CACHE.get(cache_key)  # Issue #1308
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        if val is not None and val.get("mtime") == file_mtime:
            try:
                return self.TAGS_CACHE[cache_key]["data"]
            except SQLITE_ERRORS as e:
                self.tags_cache_error(e)
                return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            print(f"Skipping file {fname}: {err}")
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:  # On Windows, bad ref to time.clock which is deprecated?
            # self.io.tool_error(f"Error lexing {fname}")
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(
        self,
        chat_fnames,
        other_fnames,
        mentioned_fnames,
        mentioned_idents,
        progress=None,
    ):
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        try:
            cache_size = len(self.TAGS_CACHE)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            cache_size = len(self.TAGS_CACHE)

        if len(fnames) - cache_size > 100:
            self.io.tool_output(
                "Initial repo scan can be slow in larger repos, but only happens once."
            )
            fnames = tqdm(fnames, desc="Scanning repo")
            showing_bar = True
        else:
            showing_bar = False

        for fname in fnames:
            if self.verbose:
                self.io.tool_output(f"Processing {fname}")
            if progress and not showing_bar:
                progress()

            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    self.io.tool_warning(f"Repo-map can't include {fname}")
                    self.io.tool_output(
                        "Has it been deleted from the file system but not from git?"
                    )
                    self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        for ident in idents:
            if progress:
                progress()

            definers = defines[ident]
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"):
                mul = 0.1
            else:
                mul = 1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            # Issue #1536
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            if progress:
                progress()

            src_rank = ranked[src]
            total_weight = sum(
                data["weight"] for _src, _dst, data in G.out_edges(src, data=True)
            )
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            # print(f"{rank:.03f} {fname} {ident}")
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(
            self.get_rel_fname(fname) for fname in other_fnames
        )

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted(
            [(rank, node) for (node, rank) in ranked.items()], reverse=True
        )
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # Create a cache key
        cache_key = [
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        ]

        if self.refresh == "auto":
            cache_key += [
                tuple(sorted(mentioned_fnames)) if mentioned_fnames else None,
                tuple(sorted(mentioned_idents)) if mentioned_idents else None,
            ]
        cache_key = tuple(cache_key)

        use_cache = False
        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            # Check if the result is in the cache
            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames,
            other_fnames,
            max_map_tokens,
            mentioned_fnames,
            mentioned_idents,
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner("Updating repo map")

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        other_rel_fnames = sorted(
            set(self.get_rel_fname(fname) for fname in other_fnames)
        )
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(max_map_tokens // 25, num_tags)
        while lower_bound <= upper_bound:
            # dump(lower_bound, middle, upper_bound)

            spin.step()

            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (
                num_tokens <= max_map_tokens and num_tokens > best_tree_tokens
            ) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        spin.end()
        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if (
            rel_fname not in self.tree_context_cache
            or self.tree_context_cache[rel_fname]["mtime"] != mtime
        ):
            code = self.io.read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                # header_max=30,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            if this_rel_fname in chat_rel_fnames:
                continue

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname
                    summary = self.get_file_summary(cur_abs_fname)
                    if summary:
                        output += f" - {summary}"
                    output += "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    # Load the tags queries
    try:
        return resources.files(__package__).joinpath(
            "queries", f"tree-sitter-{lang}-tags.scm"
        )
    except KeyError:
        return


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res


def get_file_summary(self, fname):
    """Generate a summary of the file's contents using the LLM"""
    if not self.main_model.weak_model:
        return None

    rel_fname = self.get_rel_fname(fname)
    # Skip hidden files (starting with '.')
    if os.path.basename(rel_fname).startswith("."):
        return None

    mime_type, _ = mimetypes.guess_type(fname)
    if not mime_type:
        return None

    if mime_type.startswith("image/") or "stream" in mime_type:
        return None

    if fname.endswith(".md"):
        return None

    content = self.io.read_text(fname)

    if not isinstance(content, str):
        return None

    if not content:
        return None

    # Calculate content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Initialize FileSummaryStore if needed
    if self.repo_map.file_summaries is None:
        self.repo_map.file_summaries = FileSummaryStore(self.root)

    # Try to load from persistent store
    cached = self.repo_map.file_summaries.get_summary(rel_fname, content_hash)
    if cached:
        return cached

    # Split content into chunks of 2048 characters
    chunk_size = 2048
    content_chunks = [
        content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
    ]

    messages = [
        dict(
            role="user",
            content=(
                "Analyze this file and create a searchable index entry:\n"
                f"File: {rel_fname}\n"
                "---\n"
                "1. Core purpose (15 words max)\n"
                "2. Key functions/classes (bullet list, function names with "
                "5-word description)\n"
                "3. Critical dependencies\n"
                "4. Data structures and types\n"
                "5. Integration points with other modules\n"
                "Skip tests, comments, and boilerplate. Focus on unique functionality."
                f"\n\n{content_chunks[0]}\n"
            ),
        ),
        dict(role="assistant", content="Ok."),
    ]

    original_stream_setting = self.stream
    self.stream = False

    try:

        for _ in self.send(messages, self.main_model.weak_model):
            pass
        summary = self.partial_response_content.strip()

        if len(summary) > 150:
            messages.append(
                dict(
                    role="user",
                    content=(
                        "The summary is too long. Please try again with a shorter "
                        "description less than 100 chars."
                    ),
                )
            )

        # Save to persistent store
        self.repo_map.file_summaries.save_summary(rel_fname, content_hash, summary)

        return summary
    except Exception as e:
        if self.verbose:
            self.io.tool_warning(f"Error generating summary for {fname}: {str(e)}")
        return None
    finally:
        self.stream = original_stream_setting


def get_folder_summary(self, folderpath):
    folder_summaries = []
    for fname in os.listdir(folderpath):
        full_path = os.path.join(folderpath, fname)
        if os.path.isfile(full_path):
            summary = self.get_file_summary(full_path)
            if summary:
                folder_summaries.append(summary)

    if not folder_summaries:
        return None

    folder_summary_prompt = (
        f"Given these file summaries from directory {os.path.basename(folderpath)}, create a"
        " module index:\n"
        "1. Module purpose\n"
        "2. Component hierarchy (max 3 levels)\n"
        "3. Data flow patterns\n"
        "4. Key interfaces and APIs\n"
        "5. Cross-module dependencies\n"
        "Preserve all function/class names but compress descriptions.\n\n"
        + "\n".join(folder_summaries)
    )

    messages = [
        dict(role="user", content=folder_summary_prompt),
        dict(role="assistant", content="Ok."),
    ]

    original_stream_setting = self.stream
    self.stream = False

    try:
        for _ in self.send(messages, self.main_model.weak_model):
            pass
        folder_summary = self.partial_response_content.strip()

        # Save to persistent store
        self.repo_map.file_summaries.save_folder_summary(folderpath, folder_summary)

        return folder_summary
    except Exception as e:
        if self.verbose:
            self.io.tool_warning(
                f"Error generating folder summary for {folderpath}: {str(e)}"
            )
        return None
    finally:
        self.stream = original_stream_setting


def get_arch_summary(self):

    folder_summaries = []
    for (
        folder_path,
        summary,
    ) in self.repo_map.file_summaries.folder_summaries.items():
        folder_summaries.append(f"Folder: {folder_path}\n{summary}")

    if not folder_summaries:
        return None

    arch_summary_prompt = (
        "Create a compressed but queryable codebase representation:\n"
        "1. System architecture (max 200 words)\n"
        "2. Component map (filepath -> core functions)\n"
        "3. Critical data flows\n"
        "4. External interface points\n"
        "5. Technology stack details\n"
        "Format as a hierarchical structure. Preserve ALL:\n"
        "- File paths\n"
        "- Function/class names\n"
        "- API endpoints\n"
        "- Database schemas\n"
        "- Message patterns\n\n" + "\n".join(folder_summaries)
    )

    messages = [
        dict(role="user", content=arch_summary_prompt),
        dict(role="assistant", content="Ok."),
    ]

    original_stream_setting = self.stream
    self.stream = False

    try:
        for _ in self.send(messages, self.main_model.weak_model):
            pass
        arch_summary = self.partial_response_content.strip()

        # Save to persistent store
        self.repo_map.file_summaries.save_architecture_summary(arch_summary)

        return arch_summary
    except Exception as e:
        if self.verbose:
            self.io.tool_warning(f"Error generating architecture summary: {str(e)}")
        return None
    finally:
        self.stream = original_stream_setting


if __name__ == "__main__":
    fnames = sys.argv[1:]

    chat_fnames = []
    other_fnames = []
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            chat_fnames += find_src_files(fname)
        else:
            chat_fnames.append(fname)

    rm = RepoMap(root=".")
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames)

    dump(len(repo_map))
    print(repo_map)
