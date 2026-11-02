import base64
import logging
import re
import shutil

import nbformat
import pytest

from academic import cli
from academic.import_notebook import import_notebook
from academic.utils import AcademicError

# A 1x1 transparent PNG used as stand-in content for image resources.
PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _png(payload: bytes) -> bytes:
    return PNG_BYTES + payload


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _write_notebook(folder, filename, cells):
    path = folder / filename
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    return path


def _markdown_cell(source, attachments=None):
    cell = nbformat.v4.new_markdown_cell(source)
    if attachments:
        cell["attachments"] = {name: {"image/png": _b64(data)} for name, data in attachments.items()}
    return cell


def _image_output_cell(data):
    return nbformat.v4.new_code_cell(
        "plt.show()",
        outputs=[nbformat.v4.new_output("display_data", data={"image/png": _b64(data), "text/plain": "<Figure>"})],
    )


def _bundle_references(markdown):
    """All URLs referenced from Markdown links/images, HTML tags and ref definitions, ignoring code."""
    stripped = re.sub(r"`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`", "", markdown)
    refs = set(re.findall(r"!?\[[^\]]*\]\(<?([^)\s>]+)", stripped))
    refs.update(re.findall(r"(?:src|href)=[\"']([^\"']+)", stripped, flags=re.IGNORECASE))
    refs.update(re.findall(r"^[ \t]{0,3}\[[^\]]+\]:[ \t]+(<(?:[^>\n]+)>|(?:[^ \t\n]+))", stripped, flags=re.MULTILINE))
    return refs


def _assert_self_contained(bundle):
    text = (bundle / "index.md").read_text()
    for ref in _bundle_references(text):
        ref = ref.strip("<>")
        if ref.startswith(("#", "/")) or re.match(r"^[a-z][a-z0-9+.-]*:", ref):
            continue
        name = ref.split("#", 1)[0].split("?", 1)[0].strip("<>")
        assert "/" not in name, f"reference `{ref}` was not rewritten to a flat bundle name"
        assert (bundle / name).is_file(), f"reference `{ref}` is missing from the page bundle"


def test_notebook_import_no_output(capfd):
    """
    The importer does not have a flag to determine what formats the user intends to convert.
    Instead, it relies on the file extension in the input string.
    If there is no file extension (e.g. just a wildcard), then expect no output.
    """
    cli.parse_args(["import", "--dry-run", "--verbose", "tests/data/notebooks/*", "content/post/"])
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_notebook_import_info_level(caplog):
    caplog.set_level(logging.INFO)

    cli.parse_args(
        [
            "import",
            "tests/data/notebooks/*.ipynb",
            "content/post/",
            "--dry-run",
        ]
    )
    # assert "Found notebook `test.ipynb`" in out
    assert "Searching for Jupyter notebooks in `tests/data/notebooks/*.ipynb`" in caplog.text


def test_notebook_import_debug_level(caplog):
    caplog.set_level(logging.DEBUG)

    cli.parse_args(
        [
            "import",
            "tests/data/notebooks/*.ipynb",
            "content/post/",
            "--dry-run",
        ]
    )

    # Note: this logging output should only be shown at DEBUG log level, so we set the corresponding level above
    assert "Found notebook `tests/data/notebooks/test.ipynb`" in caplog.text
    assert "Found notebook `tests/data/notebooks/blog-with-jupyter.ipynb`" in caplog.text


def test_local_references_packaged_rewritten_and_portable(tmp_path):
    source_dir = tmp_path / "source"
    for folder in ("images", "files", "assets"):
        (source_dir / folder).mkdir(parents=True)
    (source_dir / "images" / "fig.png").write_bytes(_png(b"-fig"))
    (source_dir / "files" / "data.csv").write_bytes(b"a,b\n1,2\n")
    (source_dir / "assets" / "plot.jpg").write_bytes(b"jpeg-bytes")
    (source_dir / "files" / "spec.pdf").write_bytes(b"%pdf-spec")

    notebook = _write_notebook(
        source_dir,
        "packaged-post.ipynb",
        [
            _markdown_cell(
                "# Packaged post\n\n"
                "![fig](images/fig.png)\n\n"
                "[download data](files/data.csv)\n\n"
                "[![linked plot](assets/plot.jpg)](files/data.csv)\n\n"
                '<img src="assets/plot.jpg" alt="plot">\n\n'
                "[specification][spec]\n\n"
                '[spec]: files/spec.pdf "Spec"\n'
            )
        ],
    )
    output_dir = tmp_path / "content" / "post"

    import_notebook(str(notebook), output_dir=str(output_dir))

    bundle = output_dir / "packaged-post"
    text = (bundle / "index.md").read_text()

    # Links are rewritten to flat bundle names ...
    assert "![fig](images-fig.png)" in text
    assert "[download data](files-data.csv)" in text
    assert "[![linked plot](assets-plot.jpg)](files-data.csv)" in text
    assert 'src="assets-plot.jpg"' in text
    assert "[spec]: files-spec.pdf" in text
    for old_ref in ("images/fig.png", "files/data.csv", "assets/plot.jpg", "files/spec.pdf"):
        assert old_ref not in text

    # ... the files are copied byte-for-byte ...
    assert (bundle / "images-fig.png").read_bytes() == _png(b"-fig")
    assert (bundle / "files-data.csv").read_bytes() == b"a,b\n1,2\n"
    assert (bundle / "assets-plot.jpg").read_bytes() == b"jpeg-bytes"
    assert (bundle / "files-spec.pdf").read_bytes() == b"%pdf-spec"

    # ... and the article stays complete once the source folder is moved away.
    shutil.rmtree(source_dir)
    _assert_self_contained(bundle)


def test_same_named_resources_do_not_overwrite(tmp_path):
    source_dir = tmp_path / "source"
    (source_dir / "a").mkdir(parents=True)
    (source_dir / "b").mkdir(parents=True)
    (source_dir / "a" / "logo.png").write_bytes(_png(b"-a"))
    (source_dir / "b" / "logo.png").write_bytes(_png(b"-b"))
    (source_dir / "logo.png").write_bytes(_png(b"-root"))

    notebook = _write_notebook(
        source_dir,
        "collisions.ipynb",
        [
            _markdown_cell(
                "![root](logo.png) ![a](a/logo.png) ![b](b/logo.png) ![att](attachment:logo.png)",
                attachments={"logo.png": _png(b"-att-0")},
            ),
            _markdown_cell("![att2](attachment:logo.png)", attachments={"logo.png": _png(b"-att-1")}),
            _image_output_cell(_png(b"-cell-output")),
        ],
    )

    import_notebook(str(notebook), output_dir=str(tmp_path / "out"))
    bundle = tmp_path / "out" / "collisions"
    text = (bundle / "index.md").read_text()

    expected = {
        "logo.png": _png(b"-root"),
        "a-logo.png": _png(b"-a"),
        "b-logo.png": _png(b"-b"),
        "attachment-0-logo.png": _png(b"-att-0"),
        "attachment-1-logo.png": _png(b"-att-1"),
        "output_2_0.png": _png(b"-cell-output"),
    }
    actual = {name: (bundle / name).read_bytes() for name in expected}
    assert actual == expected
    # Every same-named source survives with distinct content.
    assert len(set(actual.values())) == len(expected)

    for ref in ("logo.png", "a-logo.png", "b-logo.png", "attachment-0-logo.png", "attachment-1-logo.png", "output_2_0.png"):
        assert f"]({ref})" in text
    _assert_self_contained(bundle)


def test_missing_local_reference_errors_before_writing(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "good.png").write_bytes(_png(b"-good"))
    notebook = _write_notebook(
        source_dir,
        "broken-post.ipynb",
        [_markdown_cell("![good](good.png)\n\n![missing](images/missing.png)")],
    )

    with pytest.raises(AcademicError, match="images/missing.png"):
        import_notebook(str(notebook), output_dir=str(tmp_path / "out"))

    # No part of the page bundle may be written when validation fails.
    assert not (tmp_path / "out" / "broken-post").exists()


def test_reference_outside_resource_root_errors(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (tmp_path / "outside.png").write_bytes(_png(b"-outside"))
    notebook = _write_notebook(source_dir, "escape.ipynb", [_markdown_cell("![x](../outside.png)")])

    with pytest.raises(AcademicError, match="outside of the notebook resource folder"):
        import_notebook(str(notebook), output_dir=str(tmp_path / "out"))

    assert not (tmp_path / "out" / "escape").exists()


def test_external_anchor_and_site_root_links_left_unchanged(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    notebook = _write_notebook(
        source_dir,
        "remote-post.ipynb",
        [_markdown_cell("[web](https://example.com/a.png) " "[site](/static/logo.png) " "[mail](mailto:a@example.com) " "[anchor](#section)")],
    )

    import_notebook(str(notebook), output_dir=str(tmp_path / "out"))
    bundle = tmp_path / "out" / "remote-post"

    # No external reference is copied locally.
    assert sorted(path.name for path in bundle.iterdir()) == ["index.md"]
    text = (bundle / "index.md").read_text()
    for ref in ("https://example.com/a.png", "/static/logo.png", "mailto:a@example.com", "#section"):
        assert ref in text
