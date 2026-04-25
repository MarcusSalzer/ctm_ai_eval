from pathlib import Path


def path_collapse_user(path: Path, home: Path | None = None):
    """Do the opposite of path.expanduser(), for portability.

    Replaces the home directory prefix with '~', so paths like
    /home/user/projects/ become ~/projects/.
    """
    if home is None:
        home = Path.home()
    try:
        relative = path.relative_to(home)
        return Path("~") / relative
    except ValueError:
        # Path is not under the home directory, return as-is
        return path
