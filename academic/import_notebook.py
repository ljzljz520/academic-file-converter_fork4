import glob
import hashlib
import json
import os
import posixpath
import re
import urllib.parse
from datetime import datetime
from pathlib import Path

import nbconvert as nbc
import nbformat as nbf
import yaml
from traitlets.config import Config

from academic.jupyter_whitespace_remover import JupyterWhitespaceRemover
from academic.utils import AcademicError

# Markdown image, e.g. `![alt](images/a.png "title")`
_MARKDOWN_IMAGE_RE = re.compile(
    r"(?P<prefix>!\[(?P<text>[^\[\]]*)\]\()(?P<open><)?(?P<url>[^)\s>]+)(?(3)>)" r"(?P<suffix>[ \t]*(?:\"[^\"]*\"|'[^']*')?\))"
)

# Markdown link, e.g. `[doc](files/a.pdf)`; link text may contain one nested image,
# as in `[![alt](images/a.png)](files/a.pdf)`.
_MARKDOWN_LINK_RE = re.compile(
    r"(?P<prefix>(?<!!)\[(?P<text>[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]\()(?P<open><)?(?P<url>[^)\s>]+)(?(3)>)"
    r"(?P<suffix>[ \t]*(?:\"[^\"]*\"|'[^']*')?\))"
)

# Resource URLs in HTML tags, e.g. `<img src="...">` or `<a href='...'>`
_HTML_LINK_RE = re.compile(
    r"(?P<prefix>\b(?:src|href)[ \t]*=[ \t]*)(?P<quote>[\"']?)(?P<url>[^\s\"'<>]+)(?P=quote)",
    flags=re.IGNORECASE,
)

# Reference-style link definitions, e.g. `[label]: files/a.pdf`
_REFERENCE_DEF_RE = re.compile(
    r"(?P<prefix>^[ \t]{0,3}\[[^\]]+\]:[ \t]+)(?P<url><[^>\n]+>|[^ \t\n]+)",
    flags=re.MULTILINE,
)

# Fenced code blocks and inline code spans must never have their contents rewritten.
_CODE_TOKEN_RE = re.compile(r"(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)")


def _get_slug(text: str) -> str:
    return text.lower().replace(" ", "-")


def uniquify_attachment_references(nb) -> None:
    """
    Give every cell attachment a unique, flat name and rewrite the
    `attachment:<name>` references inside the owning cell.

    nbconvert otherwise stores attachments under their bare file name, so two
    same-named attachments silently overwrite each other and an attachment can
    clash with a file linked from Markdown or with a cell output.
    """
    used_names = set()
    for cell_index, cell in enumerate(nb.cells):
        attachments = cell.get("attachments")
        if not attachments:
            continue

        renamed = {}
        for original_name in list(attachments.keys()):
            mime_bundle = attachments.pop(original_name)
            # Mirror nbconvert's basename sanitisation while flattening the name.
            candidate = f"attachment-{cell_index}-{_flatten_resource_name(os.path.basename(original_name))}"
            unique_name = candidate
            counter = 1
            while unique_name in used_names:
                unique_name = f"{candidate}-{counter}"
                counter += 1
            used_names.add(unique_name)
            attachments[unique_name] = mime_bundle
            renamed[original_name] = unique_name

        # Replace longer names first so e.g. `attachment:a.png` cannot corrupt
        # a sibling reference called `attachment:a.png.bak`.
        source = cell.source
        for original_name in sorted(renamed, key=len, reverse=True):
            source = source.replace("attachment:" + original_name, renamed[original_name])
        cell.source = source


def import_notebook(
    input_path,
    output_dir=os.path.join("content", "post"),
    overwrite=False,
    dry_run=False,
):
    """Import blog posts from Jupyter Notebook files"""
    from academic.cli import log

    log.info(f"Searching for Jupyter notebooks in `{input_path}`")
    for filename in glob.glob(input_path, recursive=True):
        if not (filename.endswith(".ipynb") and os.path.basename(filename) != ".ipynb_checkpoints"):
            continue

        log.debug(f"Found notebook `{filename}`")

        # Read Notebook
        nb = nbf.read(open(filename, "r"), as_version=4)

        # Export Markdown
        nbc_config = Config()
        nbc_config.MarkdownExporter.preprocessors = [JupyterWhitespaceRemover]
        exporter = nbc.MarkdownExporter(config=nbc_config)
        if not dry_run:
            _export(nb, exporter, output_dir, filename, ".md", overwrite)


def _flatten_resource_name(name) -> str:
    """
    Turn a (possibly nested) relative resource path into a single safe file name
    so that every resource of the page bundle can be addressed from `index.md`.
    """
    flat = re.sub(r"[\\/]+", "-", str(name))
    flat = re.sub(r"\s+", "-", flat)
    flat = flat.strip(" .-_")
    return flat or "resource"


def _allocate_name(candidate: str, source, registry: dict) -> str:
    """
    Reserve a bundle file name for `source` (an identity tuple).

    The same source always maps to the same name, while distinct sources whose
    preferred names collide get a stable hash suffix so nothing is overwritten.
    """
    existing = registry.get(candidate)
    if existing is None:
        registry[candidate] = source
        return candidate
    if existing == source:
        return candidate

    stem, extension = os.path.splitext(candidate)
    disambiguator = hashlib.sha1(source[1].encode("utf-8")).hexdigest()[:8]
    alternative = f"{stem}.{disambiguator}{extension}"
    counter = 1
    while alternative in registry and registry[alternative] != source:
        alternative = f"{stem}.{disambiguator}-{counter}{extension}"
        counter += 1
    registry[alternative] = source
    return alternative


def _apply_outside_code(text: str, transform) -> str:
    """Run `transform` on Markdown text while leaving code spans/fences untouched."""
    parts = _CODE_TOKEN_RE.split(text)
    # With one capturing group, even indexes are text and odd indexes are code tokens.
    for index in range(0, len(parts), 2):
        if parts[index]:
            parts[index] = transform(parts[index])
    return "".join(parts)


def _sub_markdown_link(match, resolver):
    new_url = resolver(match.group("url"))
    if new_url is None:
        return match.group(0)
    # Allocated names are plain file names, so drop any angle-bracket wrapping.
    return f"{match.group('prefix')}{new_url}{match.group('suffix')}"


def _sub_html_link(match, resolver):
    new_url = resolver(match.group("url"))
    if new_url is None:
        return match.group(0)
    quote = match.group("quote")
    return f"{match.group('prefix')}{quote}{new_url}{quote}"


def _sub_reference_def(match, resolver):
    raw_url = match.group("url")
    angled = raw_url.startswith("<") and raw_url.endswith(">")
    target = raw_url[1:-1] if angled else raw_url
    new_url = resolver(target)
    if new_url is None:
        return match.group(0)
    return f"{match.group('prefix')}<{new_url}>" if angled else f"{match.group('prefix')}{new_url}"


def _rewrite_urls(body: str, resolver) -> str:
    def transform(segment: str) -> str:
        # Rewrite nested images first so the outer link is still matched afterwards.
        segment = _MARKDOWN_IMAGE_RE.sub(lambda match: _sub_markdown_link(match, resolver), segment)
        segment = _MARKDOWN_LINK_RE.sub(lambda match: _sub_markdown_link(match, resolver), segment)
        segment = _HTML_LINK_RE.sub(lambda match: _sub_html_link(match, resolver), segment)
        segment = _REFERENCE_DEF_RE.sub(lambda match: _sub_reference_def(match, resolver), segment)
        return segment

    return _apply_outside_code(body, transform)


def plan_page_resources(body: str, resources: dict, notebook_filename) -> tuple:
    """
    Plan every file the converted page bundle depends on.

    Besides the cell outputs and cell attachments already extracted by nbconvert,
    local files referenced through relative links (images, downloads, ...) are
    validated, read, assigned a unique flat bundle name and copied into the
    page bundle, while the corresponding body links are rewritten.

    Args:
        body: the Markdown produced by nbconvert
        resources: the resources mapping produced by nbconvert (its `outputs`
            dict holds cell outputs and attachments)
        notebook_filename: the source notebook path; its folder is the only
            allowed root for relative references

    Returns:
        (body, outputs): the rewritten Markdown and a mapping of bundle file
        name to file contents

    Raises:
        AcademicError: if a relative reference is missing or resolves outside
            the notebook folder. Validation happens before any file is written.
    """
    from academic.cli import log

    resource_root = Path(notebook_filename).resolve().parent

    # Registry mapping an allocated bundle name -> owning source identity.
    # Sources are either ("nbconvert", output_name) or ("local", absolute_path),
    # so equal references share an output while distinct same-named files do not.
    registry: dict = {}
    outputs: dict = {}
    claimed_names = set()
    forced_renames = {}

    # First reserve the names nbconvert already generated for cell outputs and
    # attachments (the body already links to them), flattening any odd names.
    for resource_name, data in resources.get("outputs", {}).items():
        allocated_name = _allocate_name(_flatten_resource_name(resource_name), ("nbconvert", resource_name), registry)
        claimed_names.add(allocated_name)
        outputs[allocated_name] = data
        if allocated_name != resource_name:
            forced_renames[posixpath.normpath(resource_name)] = allocated_name

    resolved_targets: dict = {}

    def resolve(url: str):
        split = urllib.parse.urlsplit(url)
        # Leave remote links, data URIs, protocol-relative links, pure anchors
        # and site-root absolute (`/...`) links untouched - they are not files
        # living next to the notebook.
        if split.scheme or split.netloc or not split.path or split.path.startswith("/"):
            return None

        normalized_path = posixpath.normpath(urllib.parse.unquote(split.path))

        if normalized_path in forced_renames:
            new_name = forced_renames[normalized_path]
        elif normalized_path in claimed_names:
            # A cell output or attachment already owns this name.
            return None
        else:
            target = (resource_root / Path(normalized_path)).resolve()
            try:
                target.relative_to(resource_root)
            except ValueError:
                raise AcademicError(
                    f"Cannot import notebook `{notebook_filename}`: reference `{url}` resolves outside of the "
                    f"notebook resource folder `{resource_root}`."
                )
            if not target.exists():
                raise AcademicError(f"Cannot import notebook `{notebook_filename}`: referenced file `{url}` does not exist (expected `{target}`).")
            if target.is_dir():
                return None

            source_key = str(target)
            new_name = resolved_targets.get(source_key)
            if new_name is None:
                relative_name = target.relative_to(resource_root).as_posix()
                new_name = _allocate_name(_flatten_resource_name(relative_name), ("local", source_key), registry)
                with open(target, "rb") as resource_file:
                    outputs[new_name] = resource_file.read()
                resolved_targets[source_key] = new_name
                log.debug(f"Packaging local resource `{target}` as `{new_name}`")

        return urllib.parse.urlunsplit(("", "", new_name, split.query, split.fragment))

    body = _rewrite_urls(body, resolve)
    return body, outputs


def _export(nb, exporter, output_dir, filename, extension, overwrite):
    from academic.cli import log

    # Determine output path for page bundle
    filename_base = Path(filename).stem
    slug = _get_slug(filename_base)
    page_bundle_path = Path(output_dir) / slug

    # Do not overwrite blog post if it already exists
    if not overwrite and os.path.isdir(page_bundle_path):
        log.debug(f"Skipping creation of `{page_bundle_path}` as it already exists. To overwrite, add the `--overwrite` argument.")
        return

    log.info(f"Importing notebook `{filename}`")

    # Check for front matter variables in notebook metadata
    if "front_matter" in nb["metadata"]:
        front_matter_from_file = dict(nb["metadata"]["front_matter"])
        log.info(f"Found front matter metadata in notebook: {json.dumps(front_matter_from_file)}")
    else:
        front_matter_from_file = {}

    # Give attachments unique names before nbconvert extracts them so that
    # same-named attachments (or an attachment and a linked file) cannot clash.
    uniquify_attachment_references(nb)

    # Convert notebook to markdown
    body, resources = exporter.from_notebook_node(nb)

    # Collect every page resource (cell outputs, attachments and local linked
    # files) and rewrite the body links. This raises before anything is written
    # if a local reference is missing or escapes the notebook folder.
    body, outputs = plan_page_resources(body, resources, filename)

    # Try to find title as top-level heading (h1), falling back to filename
    search = re.search("^#{1}(.*)", body)
    if search:
        title = search.group(1).strip()
        # Remove the h1 heading as static site generators expect the title to be defined via front matter instead.
        body = re.sub("^#{1}(.*)", "", body)
    else:
        # Fallback to using filename as title
        # Apply transformation as expect *nix-style file naming with hyphens/underscores separating words rather than spaces.
        title = filename_base.replace("-", " ").replace("_", " ").title()

    # Initialise front matter variables
    date = datetime.now().strftime("%Y-%m-%d")
    front_matter = {"title": title, "date": date}
    front_matter.update(front_matter_from_file)
    log.info(f"Generating page with title: {front_matter['title']}")

    # Unlike the Bibtex converter, we can't easily use Ruamel YAML library here as we need to output to string
    front_matter_yaml = yaml.safe_dump(front_matter, sort_keys=False, allow_unicode=True)
    # Strip final newline as our `output` will auto-add newlines below
    front_matter_yaml = front_matter_yaml.rstrip()
    # Wrap front matter variables with triple hyphens to represent Markdown front matter
    output = "\n".join(("---", front_matter_yaml, "---", clean_markdown(body)))

    # All references validated: now create the page bundle folder and write it.
    if not os.path.exists(page_bundle_path):
        os.makedirs(page_bundle_path)

    # Export notebook resources (cell outputs, attachments and local links)
    for name, data in outputs.items():
        output_filename = Path(page_bundle_path) / name
        with open(output_filename, "wb") as image_file:
            image_file.write(data)

    # Write output file
    output_filename = os.path.join(page_bundle_path, "index" + extension)
    with open(output_filename, "w") as text_file:
        text_file.write(output)


def clean_markdown(body: str) -> str:
    """
    `nbconvert` creates too much whitespace and newlines.
    Try to tidy up the output by removing multiple new lines.
    """
    return re.sub(r"\n+(?=\n)", "\n", body)
