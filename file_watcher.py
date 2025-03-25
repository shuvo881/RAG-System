import time
import os
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from indexing_main import main
class MyEventHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            file_path = event.src_path
            filename = os.path.basename(file_path)
            print(f"File CREATED: {filename}")
            main(file_path_single=file_path)

    
    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            file_path = event.src_path
            filename = os.path.basename(file_path)
            if filename.startswith('.goutputstream-') or filename.startswith('.~') or filename.endswith('~'):
                return
            print(f"File DELETED: {filename}")
    
    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            src_path = event.src_path
            dest_path = event.dest_path
            src_filename = os.path.basename(src_path)
            dest_filename = os.path.basename(dest_path)
            print(f"File MOVED: {src_filename} -> {dest_filename}")
            main(file_path_single=dest_path)
# Set up directory to watch
watch_path = "./data/raw"
print(f"Starting file watcher for directory: {watch_path}")

# Create directory if it doesn't exist
if not os.path.exists(watch_path):
    os.makedirs(watch_path)
    print(f"Created directory: {watch_path}")

# Set up and start the observer
event_handler = MyEventHandler()
observer = Observer()
observer.schedule(event_handler, watch_path, recursive=True)
observer.start()

# Keep the script running
print("File watcher is running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping file watcher...")
finally:
    observer.stop()
    observer.join()