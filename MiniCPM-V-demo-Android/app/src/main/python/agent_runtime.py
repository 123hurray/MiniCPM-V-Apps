"""Restricted Python executor for the on-device text agent."""

import builtins
import contextlib
import io
import json
import os
import pathlib
import sys
import time
import traceback


_BLOCKED_MODULES = {
    "ctypes", "multiprocessing", "socket", "subprocess", "urllib", "http",
    "ftplib", "telnetlib", "ssl",
}


def run_code(source, workspace, timeout_seconds, max_output_chars):
    root = os.path.realpath(workspace)
    os.makedirs(root, exist_ok=True)
    deadline = time.monotonic() + int(timeout_seconds)
    stdout = io.StringIO()
    stderr = io.StringIO()

    original_open = builtins.open
    original_import = builtins.__import__
    original_io_open = io.open
    original_os_open = os.open
    original_chdir = os.chdir
    original_listdir = os.listdir
    original_scandir = os.scandir
    original_remove = os.remove
    original_unlink = os.unlink
    original_rename = os.rename
    original_replace = os.replace
    original_mkdir = os.mkdir
    original_makedirs = os.makedirs
    original_rmdir = os.rmdir
    original_removedirs = os.removedirs
    original_system = os.system
    original_popen = os.popen
    process_functions = {}

    def resolve(path):
        if isinstance(path, int):
            raise PermissionError("File descriptors are not available in the Agent sandbox")
        candidate = os.path.realpath(os.path.join(root, os.fspath(path)))
        if os.path.commonpath((root, candidate)) != root:
            raise PermissionError("Path escapes the Agent workspace")
        return candidate

    def guarded_open(path, *args, **kwargs):
        return original_open(resolve(path), *args, **kwargs)

    def guarded_io_open(path, *args, **kwargs):
        return original_io_open(resolve(path), *args, **kwargs)

    def guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None:
            raise PermissionError("dir_fd is not available in the Agent sandbox")
        return original_os_open(resolve(path), flags, mode)

    def guarded_chdir(path):
        return original_chdir(resolve(path))

    def guarded_listdir(path="."):
        return original_listdir(resolve(path))

    def guarded_scandir(path="."):
        return original_scandir(resolve(path))

    def one_path(fn):
        return lambda path, *args, **kwargs: fn(resolve(path), *args, **kwargs)

    def two_paths(fn):
        return lambda src, dst, *args, **kwargs: fn(resolve(src), resolve(dst), *args, **kwargs)

    def blocked(*_args, **_kwargs):
        raise PermissionError("Operation is not available in the Agent sandbox")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in _BLOCKED_MODULES:
            raise ImportError("Module is blocked in the Agent sandbox: " + name)
        return original_import(name, globals, locals, fromlist, level)

    def trace(frame, event, arg):
        if time.monotonic() > deadline:
            raise TimeoutError("Python execution timed out")
        return trace

    try:
        builtins.open = guarded_open
        builtins.__import__ = guarded_import
        io.open = guarded_io_open
        os.open = guarded_os_open
        os.chdir = guarded_chdir
        os.listdir = guarded_listdir
        os.scandir = guarded_scandir
        os.remove = one_path(original_remove)
        os.unlink = one_path(original_unlink)
        os.rename = two_paths(original_rename)
        os.replace = two_paths(original_replace)
        os.mkdir = one_path(original_mkdir)
        os.makedirs = one_path(original_makedirs)
        os.rmdir = one_path(original_rmdir)
        os.removedirs = one_path(original_removedirs)
        os.system = blocked
        os.popen = blocked
        for name in (
            "execv", "execve", "execl", "execle", "execlp", "execlpe", "execvp", "execvpe",
            "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        ):
            if hasattr(os, name):
                process_functions[name] = getattr(os, name)
                setattr(os, name, blocked)

        original_chdir(root)
        sys.settrace(trace)
        globals_dict = {"__name__": "__main__", "__builtins__": builtins}
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(source, "<agent>", "exec"), globals_dict, globals_dict)
        status = "ok"
    except BaseException:
        traceback.print_exc(file=stderr)
        status = "error"
    finally:
        sys.settrace(None)
        builtins.open = original_open
        builtins.__import__ = original_import
        io.open = original_io_open
        os.open = original_os_open
        os.chdir = original_chdir
        os.listdir = original_listdir
        os.scandir = original_scandir
        os.remove = original_remove
        os.unlink = original_unlink
        os.rename = original_rename
        os.replace = original_replace
        os.mkdir = original_mkdir
        os.makedirs = original_makedirs
        os.rmdir = original_rmdir
        os.removedirs = original_removedirs
        os.system = original_system
        os.popen = original_popen
        for name, function in process_functions.items():
            setattr(os, name, function)
        original_chdir(root)

    out = stdout.getvalue()
    err = stderr.getvalue()
    limit = int(max_output_chars)
    if len(out) > limit:
        out = out[:limit] + "\n…(truncated)"
    if len(err) > limit:
        err = err[:limit] + "\n…(truncated)"
    return json.dumps({"status": status, "stdout": out, "stderr": err}, ensure_ascii=False)
