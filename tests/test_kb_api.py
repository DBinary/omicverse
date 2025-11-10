import io
import os
import sys
from types import SimpleNamespace

import pytest

from omicverse.alignment import kb_api


def test_ensure_dir_creates_dir(tmp_path):
    d = tmp_path / "nested" / "dir"
    assert not d.exists()
    kb_api._ensure_dir(str(d))
    assert d.exists()
    assert d.is_dir()


def test_append_flag_scalar_and_bool():
    cmd = []
    kb_api._append_flag(cmd, '--opt', 'value')
    assert '--opt' in cmd and 'value' in cmd

    cmd = []
    kb_api._append_flag(cmd, '--flag', True, as_bool=True)
    assert '--flag' in cmd

    cmd = []
    kb_api._append_flag(cmd, '--flag-false', False, as_bool=True)
    assert '--flag-false' not in cmd


def test_normalize_list_arg():
    assert kb_api._normalize_list_arg(None) is None
    assert kb_api._normalize_list_arg('a,b') == 'a,b'
    assert kb_api._normalize_list_arg(['a', 'b']) == 'a,b'
    assert kb_api._normalize_list_arg(['x', 'y'], sep=';') == 'x;y'


def test_include_exclude_to_flags_various_forms():
    include = [
        {"attribute": "gene", "pattern": "MT"},
        {"key": "transcript", "value": "ENST"}
    ]
    exclude = [
        {"attribute": "junk", "pattern": "foo"},
        {"key": "bad", "value": "bar"}
    ]
    flags = kb_api._include_exclude_to_flags(include, exclude)
    # Expect pairs: --include-attribute gene:MT --include-attribute transcript:ENST etc.
    assert '--include-attribute' in flags
    assert any('gene:MT' in x for x in flags) or any(x == 'gene:MT' for x in flags)
    assert '--exclude-attribute' in flags
    assert any('bad:bar' in x for x in flags) or any(x == 'bad:bar' for x in flags)


class DummyPopen:
    """
    Simulate subprocess.Popen for tests. Use io.StringIO for stdout so that it is
    both iterable and provides close(), matching real file-like object behavior.
    """
    def __init__(self, cmd, stdout_lines=None, returncode=0, **kwargs):
        self.cmd = cmd
        if stdout_lines is None:
            stdout_lines = ["line1\n", "line2\n"]
        # StringIO is iterable (yields lines) and has close()
        self.stdout = io.StringIO(''.join(stdout_lines))
        self._returncode = returncode

    def wait(self):
        return self._returncode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ensure close behavior if used as context manager
        try:
            self.stdout.close()
        except Exception:
            pass


def test_run_kb_success_monkeypatch(monkeypatch, capsys):
    # Patch subprocess.Popen to our DummyPopen with returncode 0
    def fake_popen(cmd, stdout, stderr, cwd, env, text, bufsize):
        return DummyPopen(cmd, stdout_lines=["ok\n"], returncode=0)

    monkeypatch.setattr(kb_api.subprocess, "Popen", fake_popen)

    # Call _run_kb; should not raise
    kb_api._run_kb(["echo", "hello"])
    captured = capsys.readouterr()
    assert ">>" in captured.out  # the command print
    assert "ok" in captured.out  # the line printed from stdout


def test_run_kb_failure_raises(monkeypatch):
    def fake_popen(cmd, stdout, stderr, cwd, env, text, bufsize):
        return DummyPopen(cmd, stdout_lines=["err\n"], returncode=2)

    monkeypatch.setattr(kb_api.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError):
        kb_api._run_kb(["cmd", "will_fail"])


def test_ref_builds_cmd_and_calls_run_kb(monkeypatch, tmp_path):
    # deterministic uuid4 for predictable tmp dir name
    class DummyUUID:
        hex = "deadbeef"

    monkeypatch.setattr(kb_api, "uuid4", lambda: DummyUUID())

    # Ensure _which_kb returns a simple token and _run_kb captures the cmd
    monkeypatch.setattr(kb_api, "_which_kb", lambda: "kb_exec")
    recorded = {}

    def fake_run_kb(cmd, env=None, cwd=None):
        recorded['cmd'] = list(cmd)
        return None

    monkeypatch.setattr(kb_api, "_run_kb", fake_run_kb)

    index_path = str(tmp_path / "out" / "index.idx")
    t2g_path = str(tmp_path / "out" / "t2g.txt")
    res = kb_api.ref(index_path=index_path, t2g_path=t2g_path, d="prebuilt", cdna_path=str(tmp_path / "cdna.fa"), threads=4)
    assert 'cmd' in recorded
    cmd = recorded['cmd']
    # -d should be present and followed by our value
    assert '-d' in cmd
    idx = cmd.index('-d')
    assert cmd[idx + 1] == "prebuilt"
    # -i and -g should be present
    assert '-i' in cmd and index_path in cmd
    assert '-g' in cmd and t2g_path in cmd
    # result dict must include index_path and t2g_path
    assert res['index_path'] == index_path
    assert res['t2g_path'] == t2g_path


def test_count_builds_cmd_and_calls_run_kb(monkeypatch, tmp_path):
    monkeypatch.setattr(kb_api, "_which_kb", lambda: "kb_exec")
    recorded = {}

    def fake_run_kb(cmd, env=None, cwd=None):
        recorded['cmd'] = list(cmd)
        return None

    monkeypatch.setattr(kb_api, "_run_kb", fake_run_kb)

    index_path = str(tmp_path / "ix.idx")
    t2g_path = str(tmp_path / "t2g.txt")
    fastq = [str(tmp_path / "r1.fastq"), str(tmp_path / "r2.fastq")]
    out = kb_api.count(index_path=index_path, t2g_path=t2g_path, technology="10XV3", fastq_paths=fastq, output_path=str(tmp_path / "out"))
    assert 'cmd' in recorded
    cmd = recorded['cmd']
    # check presence of required flags
    assert '-x' in cmd and '10XV3' in cmd
    assert '-i' in cmd and index_path in cmd
    assert '-g' in cmd and t2g_path in cmd
    # fastq inputs appended
    assert fastq[0] in cmd and fastq[1] in cmd
    assert out['technology'] == '10XV3'
    assert out['output_path'] == str(tmp_path / "out")


def test_analyze_10x_v3_data_calls_ref_and_count(monkeypatch, tmp_path):
    recorded = {}
    # stub ref and count to avoid invoking _run_kb
    def fake_ref(**kwargs):
        recorded['ref'] = kwargs
        return {'index_path': str(tmp_path / 'ref' / 'index.idx'), 't2g_path': str(tmp_path / 'ref' / 't2g.txt'), 'workflow': 'nucleus'}

    def fake_count(**kwargs):
        recorded['count'] = kwargs
        return {'workflow': 'nucleus', 'technology': '10XV3', 'output_path': kwargs.get('output_path')}

    monkeypatch.setattr(kb_api, "ref", fake_ref)
    monkeypatch.setattr(kb_api, "count", fake_count)

    fastqs = [str(tmp_path / "r1.fastq")]
    res = kb_api.analyze_10x_v3_data(fastq_files=fastqs, reference_output_dir=str(tmp_path / "reference"), analysis_output_dir=str(tmp_path / "analysis"), download_reference=True, threads_ref=2, threads_count=1)
    assert 'reference' in res and 'count' in res
    assert 'ref' in recorded and 'count' in recorded