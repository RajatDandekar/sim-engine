"""SimEngine entry point: python -m sim_engine <command>

Commands:
    app      — Launch the full UI (record, train, evaluate — all in one)
    view     — Simple demo viewer
    record   — Record episodes (CLI)
    train    — Train a policy (CLI)
    eval     — Evaluate a policy (CLI)
"""

import sys


def main():
    if len(sys.argv) < 2:
        # Default to app
        from .app import main as app_main
        app_main()
        return

    command = sys.argv[1]
    sys.argv = sys.argv[1:]

    if command == "app":
        from .app import main as app_main
        app_main()
    elif command == "view":
        from .viewer import main as viewer_main
        viewer_main()
    elif command == "record":
        from .record import main as record_main
        record_main()
    elif command == "train":
        from .train import main as train_main
        train_main()
    elif command == "eval":
        from .evaluate import main as eval_main
        eval_main()
    else:
        print(f"Unknown command: {command}")
        print("Available: app, view, record, train, eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
