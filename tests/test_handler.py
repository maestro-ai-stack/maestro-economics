"""Tests for runtime.handler — job execution engine."""

import json
import os
import tempfile

import pytest

from runtime.handler import execute_job


class TestHandler:
    def test_executes_run_ctx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write('def run(ctx):\n    return {"answer": 42}\n')
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            result = execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=os.path.join(tmpdir, "output"),
                config={},
                progress_cb=lambda p, m: None,
            )
            assert result["answer"] == 42

    def test_captures_progress(self):
        calls: list[tuple[float, str]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write(
                    'def run(ctx):\n'
                    '    ctx.progress(0.5, "half")\n'
                    '    return {}\n'
                )
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=os.path.join(tmpdir, "output"),
                config={},
                progress_cb=lambda p, m: calls.append((p, m)),
            )
        assert (0.5, "half") in calls

    def test_missing_run_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write("x = 1\n")
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            with pytest.raises(ValueError, match="must define a run"):
                execute_job(
                    script_path=script,
                    data_dir=data_dir,
                    output_dir=os.path.join(tmpdir, "output"),
                    config={},
                    progress_cb=lambda p, m: None,
                )

    def test_output_files_collected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write(
                    "import os\n"
                    "def run(ctx):\n"
                    '    with open(os.path.join(str(ctx.output_dir), "table.tex"), "w") as f:\n'
                    '        f.write("\\\\begin{table}\\\\end{table}")\n'
                    '    return {"done": True}\n'
                )
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            out_dir = os.path.join(tmpdir, "output")

            execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=out_dir,
                config={},
                progress_cb=lambda p, m: None,
            )
            assert os.path.exists(os.path.join(out_dir, "table.tex"))

    def test_results_json_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write('def run(ctx):\n    return {"beta": 1.5}\n')
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            out_dir = os.path.join(tmpdir, "output")

            execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=out_dir,
                config={},
                progress_cb=lambda p, m: None,
            )
            with open(os.path.join(out_dir, "results.json")) as f:
                saved = json.load(f)
            assert saved["beta"] == 1.5

    def test_stdout_captured_in_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write(
                    'def run(ctx):\n'
                    '    print("hello from job")\n'
                    '    return {}\n'
                )
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            out_dir = os.path.join(tmpdir, "output")

            execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=out_dir,
                config={},
                progress_cb=lambda p, m: None,
            )
            with open(os.path.join(out_dir, "job.log")) as f:
                log = f.read()
            assert "hello from job" in log

    def test_invalid_script_path_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Cannot load script"):
                execute_job(
                    script_path=os.path.join(tmpdir, "nonexistent.py"),
                    data_dir=tmpdir,
                    output_dir=os.path.join(tmpdir, "output"),
                    config={},
                    progress_cb=lambda p, m: None,
                )

    def test_non_dict_result_wrapped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write('def run(ctx):\n    return "just a string"\n')
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            result = execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=os.path.join(tmpdir, "output"),
                config={},
                progress_cb=lambda p, m: None,
            )
            assert result["output"] == "just a string"

    def test_gpu_info_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w") as f:
                f.write(
                    'def run(ctx):\n'
                    '    return {"gpu_type": ctx.gpu.type, "gpu_mem": ctx.gpu.memory_gb}\n'
                )
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            result = execute_job(
                script_path=script,
                data_dir=data_dir,
                output_dir=os.path.join(tmpdir, "output"),
                config={},
                progress_cb=lambda p, m: None,
                gpu_type="RTX 4090",
                gpu_memory_gb=24,
            )
            assert result["gpu_type"] == "RTX 4090"
            assert result["gpu_mem"] == 24
