from run_storage import regenerate_master_log


if __name__ == "__main__":
    path = regenerate_master_log()
    print(f"Updated: {path}")
