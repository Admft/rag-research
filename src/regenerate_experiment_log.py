from experiment_summary import regenerate_experiment_log


if __name__ == "__main__":
    path = regenerate_experiment_log()
    print(f"Updated: {path}")
