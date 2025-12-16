import subprocess

def test_flake8_linter():
    """Test if flake8 linter runs without issues."""
    result = subprocess.run(['flake8', '--max-line-length=120'], capture_output=True, text=True)
    assert result.returncode == 0, f"Linting failed:\n{result.stdout}"

def test_pytest():
    """Test if pytest runs all unit tests successfully."""
    result = subprocess.run(['pytest', '--maxfail=1', '--disable-warnings', '-q'], capture_output=True, text=True)
    assert result.returncode == 0, f"Tests failed:\n{result.stdout}"
