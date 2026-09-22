"""Child process entry point for persistent jobs."""
import sys
from .jobs import work
if __name__ == '__main__':
    work(sys.argv[1])

