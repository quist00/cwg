import os
import sys


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    backend_src = os.path.join(repo_root, 'backend', 'src')
    if backend_src not in sys.path:
        sys.path.insert(0, backend_src)

    import gen  # noqa: E402
    # Forward all arguments directly to the original CLI
    gen.main(sys.argv[1:])


if __name__ == '__main__':
    main()
