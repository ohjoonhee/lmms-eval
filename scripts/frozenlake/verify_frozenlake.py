import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from lmms_eval.tasks import task_manager

print("Checking if frozenlake_vqa task loads...")
try:
    tm = task_manager.TaskManager()
    tasks = tm.load_task_dict(["frozenlake_vqa"])
    print("Tasks loaded:", list(tasks.keys()))

    # Check if dataset loads
    task = tasks["frozenlake_vqa"]
    print("Task instance:", task)
    # Trigger download/parsing if lazy
    # task.download() # Might not be available on instance, but loading dict usually initializes it.

except Exception as e:
    print("Error loading task:", e)
    import traceback

    traceback.print_exc()
