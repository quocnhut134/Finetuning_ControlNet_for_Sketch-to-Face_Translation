import os
import sys
import subprocess

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    app_file_path = os.path.join(project_dir, "streamlit_app.py")
    subprocess.run(["streamlit", "run", app_file_path], cwd=project_dir)

if __name__ == "__main__":
    main()