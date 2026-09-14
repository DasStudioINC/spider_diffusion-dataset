import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import queue
import os
import time
from PIL import Image, ImageTk
import glob

class SpiderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spider AI Image Generator UI")
        self.geometry("1100x700")
        self.configure(bg="#121212")

        self.commands_help = {
            "python train_diffusion.py --epochs": "Train model (Args: <int> e.g. 1000)",
            "python generate_image.py --prompt": "Generate image from text (Args: \"your prompt here\")",
            "clear": "Clear the terminal console screen",
            "help": "List all available interactive tools and commands"
        }

        self.setup_styles()
        self.create_header()
        self.create_main_split()
        self.create_command_bar()

        self.output_queue = queue.Queue()
        self.process = None
        self.is_running = False
        self.start_time = 0
        
        self.active_preview_path = None
        self.last_modified_time = 0
        self.current_image_index = 0
        self.cached_images_list = []
        self.last_command_type = "train" # Tracks whether we were "train" or "generate"

        # Command history tracking variables
        self.command_history = []
        self.history_index = -1

        self.after(100, self.poll_subprocess_output)
        self.after(500, self.update_status_and_timer)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
    def create_header(self):
        header_frame = tk.Frame(self, bg="#1e1e1e", height=45)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_title = tk.Label(header_frame, text="Spider AI Image Generator UI (PYTHON)", bg="#990000", fg="#ffffff", font=("Consolas", 10, "bold"), padx=10, pady=5)
        self.lbl_title.pack(side=tk.LEFT, padx=5)
        self.lbl_version = tk.Label(header_frame, text="Version 1.0.0", bg="#222222", fg="#ffffff", font=("Consolas", 10, "bold"), padx=10, pady=5)
        self.lbl_version.pack(side=tk.LEFT, padx=5)
        self.lbl_model_size = tk.Label(header_frame, text="Model Size: -- MB", bg="#222222", fg="#ff4444", font=("Consolas", 10, "bold"), padx=10, pady=5)
        self.lbl_model_size.pack(side=tk.LEFT, padx=5)
        self.lbl_status = tk.Label(header_frame, text="Status: IDLE", bg="#222222", fg="#b58900", font=("Consolas", 10, "bold"), padx=10, pady=5)
        self.lbl_status.pack(side=tk.LEFT, padx=5)
        self.lbl_time = tk.Label(header_frame, text="Time: 00:00", bg="#222222", fg="#ffffff", font=("Consolas", 10, "bold"), padx=10, pady=5)
        self.lbl_time.pack(side=tk.LEFT, padx=5)

    def create_main_split(self):
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#121212", sashwidth=6)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 0))

        left_container = tk.Frame(main_pane, bg="#121212")
        main_pane.add(left_container, width=540)
        self.terminal_box = tk.Text(left_container, bg="#1a1a1a", fg="#00ff66", font=("Consolas", 10), insertbackground="white", state=tk.DISABLED)
        self.terminal_box.pack(fill=tk.BOTH, expand=True)

        right_container = tk.Frame(main_pane, bg="#1e1e1e")
        main_pane.add(right_container, width=540)
        preview_title = tk.Label(right_container, text="Live Training Visualization / Generation Output", bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 10, "bold"))
        preview_title.pack(pady=10)
        self.image_canvas = tk.Canvas(right_container, bg="#151515", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_text_id = self.image_canvas.create_text(250, 220, text="Waiting for training or generation...", fill="#666666", font=("Consolas", 10), justify=tk.CENTER)

    def create_command_bar(self):
        bottom_wrapper = tk.Frame(self, bg="#121212")
        bottom_wrapper.pack(fill=tk.X, padx=10, pady=10)
        self.suggestion_listbox = tk.Listbox(bottom_wrapper, bg="#222222", fg="#ffcc00", font=("Consolas", 9), height=4)
        self.suggestion_listbox.bind("<<ListboxSelect>>", self.apply_autocomplete)
        bar_frame = tk.Frame(bottom_wrapper, bg="#1e1e1e", height=50)
        bar_frame.pack(fill=tk.X, pady=(2, 0))
        lbl = tk.Label(bar_frame, text="CMD>", bg="#1e1e1e", fg="#ff4444", font=("Consolas", 11, "bold"))
        lbl.pack(side=tk.LEFT, padx=5)
        self.cmd_entry = tk.Entry(bar_frame, bg="#151515", fg="#ffffff", font=("Consolas", 11), insertbackground="white")
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=8)
        self.cmd_entry.bind("<KeyRelease>", self.on_command_key_release)
        self.cmd_entry.bind("<Return>", self.execute_user_command)
        
        # Bind Up and Down arrow keys for history navigation
        self.cmd_entry.bind("<Up>", self.on_key_up)
        self.cmd_entry.bind("<Down>", self.on_key_down)

    def on_command_key_release(self, event):
        typed_text = self.cmd_entry.get().strip()
        if not typed_text:
            self.suggestion_listbox.pack_forget()
            return
        matches = [cmd for cmd in self.commands_help.keys() if cmd.startswith(typed_text) or typed_text in cmd]
        if matches:
            self.suggestion_listbox.delete(0, tk.END)
            for m in matches:
                self.suggestion_listbox.insert(tk.END, f"{m}  -->  {self.commands_help[m]}")
            self.suggestion_listbox.pack(fill=tk.X, before=self.cmd_entry.master, pady=(0, 2))
        else:
            self.suggestion_listbox.pack_forget()

    def apply_autocomplete(self, event):
        selection = self.suggestion_listbox.curselection()
        if selection:
            selected_text = self.suggestion_listbox.get(selection[0])
            base_cmd = selected_text.split("  -->  ")[0]
            self.cmd_entry.delete(0, tk.END)
            self.cmd_entry.insert(0, base_cmd + " ")
            self.suggestion_listbox.pack_forget()
            self.cmd_entry.focus()

    def on_key_up(self, event):
        """Navigate backwards through command history."""
        if not self.command_history:
            return
        if self.history_index > 0:
            self.history_index -= 1
        elif self.history_index == -1:
            self.history_index = len(self.command_history) - 1
        
        self.cmd_entry.delete(0, tk.END)
        self.cmd_entry.insert(0, self.command_history[self.history_index])
        return "break" # Prevents default text cursor behavior

    def on_key_down(self, event):
        """Navigate forwards through command history."""
        if not self.command_history:
            return
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.cmd_entry.delete(0, tk.END)
            self.cmd_entry.insert(0, self.command_history[self.history_index])
        else:
            self.history_index = len(self.command_history)
            self.cmd_entry.delete(0, tk.END) # Clear if we go past the newest command
        return "break"

    def log_to_terminal(self, message):
        self.terminal_box.config(state=tk.NORMAL)
        self.terminal_box.insert(tk.END, message + "\n")
        self.terminal_box.see(tk.END)
        self.terminal_box.config(state=tk.DISABLED)

    def execute_user_command(self, event):
        command = self.cmd_entry.get().strip()
        if not command: return
        self.suggestion_listbox.pack_forget()
        self.log_to_terminal(f"> {command}")
        self.cmd_entry.delete(0, tk.END)

        # Add to history and reset pointer position
        self.command_history.append(command)
        self.history_index = len(self.command_history)

        if command == "clear":
            self.terminal_box.config(state=tk.NORMAL)
            self.terminal_box.delete("1.0", tk.END)
            self.terminal_box.config(state=tk.DISABLED)
            return
        
        # Track whether this command is training or generating
        if "train" in command.lower():
            self.last_command_type = "train"
        elif "generate" in command.lower():
            self.last_command_type = "generate"

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.is_running = True
        self.start_time = time.time()
        self.lbl_status.config(text="Status: RUNNING", fg="#00ff66")
        self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        threading.Thread(target=self.enqueue_output, args=(self.process.stdout, self.output_queue), daemon=True).start()

    def enqueue_output(self, out, queue):
        for line in iter(out.readline, ''):
            queue.put(line)
        out.close()
        queue.put("--- PROCESS FINISHED ---")

    def poll_subprocess_output(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "--- PROCESS FINISHED ---":
                    self.is_running = False
                    self.lbl_status.config(text="Status: IDLE", fg="#b58900")
                else:
                    self.log_to_terminal(line.rstrip())
        except queue.Empty: pass
        self.after(100, self.poll_subprocess_output)

    def update_status_and_timer(self):
        if self.is_running:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.lbl_time.config(text=f"Time: {mins:02d}:{secs:02d}")
        else:
            self.lbl_time.config(text="Time: 00:00")
        ckpt_path = "diffusion_checkpoint.pth"
        if os.path.exists(ckpt_path):
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            self.lbl_model_size.config(text=f"Model Size: {size_mb:.1f} MB")
        else:
            self.lbl_model_size.config(text="Model Size: 0.0 MB")
        self.check_for_preview_images()
        self.after(500, self.update_status_and_timer)

    def check_for_preview_images(self):
        """
        Dynamically handles canvas previews based on what you are doing:
        - If training (or training was last run), cycles through training images (spe1...spe4).
        - If generating (or generation was last run), shows the latest generated output image.
        """
        base_cache_dir = "./cache_images"
        expected_files = [f"spe{i}.png" for i in range(1, 5)]
        self.cached_images_list = [os.path.join(base_cache_dir, f) for f in expected_files if os.path.exists(os.path.join(base_cache_dir, f))]

        # If training was the last action, prioritize cycling through dataset images
        if self.last_command_type == "train" and self.cached_images_list:
            cycle_interval = 5
            self.current_image_index = int(time.time() % (len(self.cached_images_list) * cycle_interval)) // cycle_interval
            if self.current_image_index < len(self.cached_images_list):
                self.load_canvas_image(self.cached_images_list[self.current_image_index])
                return

        # If generating was the last action, show the latest generated output image
        if self.last_command_type == "generate":
            all_pngs = [f for f in glob.glob("*.png") if not f.startswith("spe")]
            if all_pngs:
                latest_gen_image = max(all_pngs, key=os.path.getmtime)
                self.load_canvas_image(latest_gen_image)
                return

        # Fallback to training cache if available
        if self.cached_images_list:
            cycle_interval = 5
            self.current_image_index = int(time.time() % (len(self.cached_images_list) * cycle_interval)) // cycle_interval
            if self.current_image_index < len(self.cached_images_list):
                self.load_canvas_image(self.cached_images_list[self.current_image_index])

    def load_canvas_image(self, path):
        """Helper to safely load and scale an image to the canvas."""
        if self.active_preview_path == path:
            mtime = os.path.getmtime(path)
            if mtime == self.last_modified_time:
                return 
            self.last_modified_time = mtime

        try:
            img = Image.open(path)
            img = img.resize((350, 350), Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(200, 200, image=self.photo, anchor=tk.CENTER)
            self.active_preview_path = path
            self.last_modified_time = os.path.getmtime(path)
        except Exception:
            pass 

if __name__ == "__main__":
    app = SpiderUI()
    app.mainloop()