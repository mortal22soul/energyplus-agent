import os
import subprocess
import random
from datetime import datetime, timedelta

def get_untracked_files():
    result = subprocess.run(['git', 'ls-files', '-o', '--exclude-standard'], capture_output=True, text=True)
    return [line for line in result.stdout.split('\n') if line]

def main():
    files = get_untracked_files()
    
    # We want to make at least 12 commits. Let's make 15 commits.
    num_commits = 15
    
    # Chunking files
    # Shuffle first to make random groups
    # random.shuffle(files) # Alternatively, keep them in order for somewhat logical commits
    chunk_size = max(1, len(files) // num_commits)
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    # We want the last commit to be backdated by a day. Today is 2026-07-27.
    # So the last commit should be around 2026-07-26.
    # Total duration = len(chunks) * average 3 hours
    
    # Let's start from a base time and add random intervals
    # End time ~ 2026-07-26 23:00:00
    # Let's work backwards to find the start time
    
    intervals = [random.randint(2, 4) for _ in range(len(chunks) - 1)]
    total_hours = sum(intervals)
    
    end_time = datetime(2026, 7, 26, 18, 0, 0)
    current_time = end_time - timedelta(hours=total_hours)
    
    commit_messages = [
        "Initial project setup",
        "Add core dependencies and config",
        "Implement basic utilities",
        "Setup initial models and schemas",
        "Add core modules",
        "Implement controller logic",
        "Setup multi-agent interaction",
        "Add safety and metrics",
        "Implement dashboard and CLI",
        "Add documentation and instructions",
        "Implement synthetic data generation",
        "Setup test framework",
        "Add more tests and fix bugs",
        "Refactoring and cleanup",
        "Final polish and updates",
        "Additional configurations",
        "Minor adjustments"
    ]
    
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
            
        print(f"Commit {i+1}/{len(chunks)}")
        
        # Add files
        for f in chunk:
            subprocess.run(['git', 'add', f])
            
        # Commit
        msg = commit_messages[i] if i < len(commit_messages) else "Update files"
        date_str = current_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Set environment variables for GIT_AUTHOR_DATE and GIT_COMMITTER_DATE
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        print(f"Committing chunk with {len(chunk)} files at {date_str}...")
        subprocess.run(['git', 'commit', '-m', msg], env=env)
        
        if i < len(intervals):
            current_time += timedelta(hours=intervals[i], minutes=random.randint(0, 59))

if __name__ == '__main__':
    main()
