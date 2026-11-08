# Safe workspace edits

This is the Package B file-editing design as implemented in `workspace.py` and
`patches.py`. It does not change provider, transport, or session behavior.
Chat history is not updated by these modules.

## What you can do

Stage regular UTF-8 files under one workspace root, preview a model proposal,
and apply it only after an unambiguous `y`. Undo restores the backup the same
way.

```text
/file root .
/file add src/app.py tests/test_app.py
/file list
/file show src/app.py
/file remove src/app.py
/file clear
/edit --format diff "Handle empty input in src/app.py"
/edit --format blocks "Rename the helper in src/app.py"
/edit show
/edit apply
/edit discard
/undo list
/undo last
```

`/file add` only copies nothing and sends nothing. It records path, size,
sha256, permissions, and newline style in the current process.

`/edit` without a connected proposal runner does not call a provider. Pass
model text with the handler's `model_text` argument, or connect a runner that
returns text. The runner is invoked once, only after an interactive approval
of the exact paths, byte sizes, and destination. `n`, `yes`, and an empty
answer do not approve.

Noninteractive calls print a proposal and do not write source files. They
also do not call the runner, so a pipe cannot send attachments or apply a
patch. Apply and undo exist only on the interactive path, and each one asks
again.

## Limits and refusals

Defaults, adjustable by constructing `Limits`:

- 256 KiB per file
- 1 MiB total
- 20 files

Reads stop at the byte cap. Paths must be relative, with no `..`, NUL,
backslash, colon, drive, UNC, or device prefix. The walk does not follow
symlinks. Directories, sockets, fifos, binary files, and invalid UTF-8 are
refused.

These names are refused before the body is treated as an attachment: `.env`
and `.env.*`, `keys.json`, `credentials.json`, private-key filenames, and
`.pem`, `.key`, `.p12`, `.pfx`, `.kdbx`. Content that looks like a private
key, an AWS access key id, or a credential assignment is flagged. The flag
does not include the matched text. Detection is incomplete.

Changing the workspace root clears attachments and send approval.

## Proposals

Only attached files can appear. Mixed prose is rejected. A single surrounding
fence is accepted when it contains nothing else.

Unified diffs must be LF, with matching `---` and `+++` paths and hunk counts
that match the body. Context must match the file exactly. Overlapping hunks,
renames, `/dev/null`, and mode changes are rejected.

Search/replace input must be a JSON object whose only key is `edits`. Each
edit has only `path`, `search`, and `replace`. Each search must match once,
and ranges must not overlap.

Files with mixed newline styles are not rewritten. LF and CRLF files keep
their newline style. A missing final newline is preserved when the diff says
`\ No newline at end of file`, and is not invented or dropped silently.

The preview is a new diff built with `difflib` from the original and proposed
text. Terminal escapes in paths and text are escaped for display only. Bytes
written to disk are the proposed file bytes, not the escaped preview.

## Apply, undo, recovery

Apply re-reads each file, checks the stored hash and permission bits, and
refuses if either changed. Backups and a versioned journal are written under
`ANYLLM_HOME/edits/<run-id>/` with owner-only permissions where the platform
allows it. If a backup cannot be created, no source file is modified.

Replacements are written to a temporary file in the destination directory and
moved with `os.replace`, then the previous permission bits are applied again.
If a later file fails, files already replaced are restored from backup when
that restore succeeds. If a restore fails, the journal is marked `partial`
and the error names the files. Multi-file atomicity is not promised.

`/undo last` restores only when the current bytes still match the applied
version. A newer edit is left in place.

Startup code may report journals in `prepared`, `applying`, or `partial`. It
does not roll anything back. Confirming recovery of a `prepared` journal only
marks it aborted. Confirming `applying` or `partial` restores files that
still match the applied bytes.

Backups can contain source, including secrets that were in that source. They
are not uploaded, pruned, or deleted by this version.

## Package A seam

`feat/orchestration` (pull request 3) was read remotely and is not merged here.
`workspace_seam.py` calls that contract when `anyllm.seam` is installed, and
the tests use a fake with the same names. On this branch the import is absent,
so `/file` and `/edit` keep working through `WorkspaceService` alone.

After consent, each file is `attachment_store().approve(ApprovedAttachment(path,
sha256, content))`. `content` is the exact UTF-8 text. The store is not
cleared wholesale, because it is process-wide. Paths this feature published
are `revoke`d when the attachment set changes.

A proposal calls:

```text
run_request(provider, target, snapshot, options, *, credential, prompt, messages=None, cancel_event=None, event_queue=None)
```

`make_snapshot(..., attachments=approved, record_refs=False)` copies history
and does not receive the API key. `RunResult.text` is the proposal only when
`status` is `ok`. `Session.append_exchange` is not used. `Presentation.block`
and `status_line` are used only by the registered REPL handler, on the main
thread.

The later wiring, after an approved merge, is `wiring_notes()` in
`workspace_seam.py`. It is one `register_workspace(register_repl)` call beside
`install_repl_features`, plus `repl.presentation`. This branch does not edit
`repl.py` or `cli.py` for that.

## What this is not

- Not a sandbox against another process that replaces directories or files
  while the apply is running.
- Not model-directed execution. Proposal text is parsed as a diff or as the
  JSON envelope above. It is never evaluated or passed to a shell.
- Not an automatic test runner. Nothing in the workspace is imported or
  executed.
- Not a live API client. No provider is contacted unless a caller passes a
  runner, and the runner is not called from noninteractive mode.

## Tests

From the `anyllm/` directory, with no API key and no network:

```sh
python3 -m unittest tests.test_workspace tests.test_patches
python3 -m unittest discover -s tests
python3 -m compileall anyllm tests
```

`ANYLLM_HOME` should point at an empty temporary directory in CI. The tests
create their own directories under `.test-run/`.
