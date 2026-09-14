import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import queue
import os
import time
import shutil
from PIL import Image, ImageTk
import glob

class SpiderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spider AI Image Generator UI")
        self.geometry("1100x700")
        self.configure(bg="#121212")

        self.commands_help = {
            "train_diffusion.py --epochs ... --pass ...": "Train model (Requires --pass <password>)",
            "generate_image.py --prompt": "Generate image from text (Args: \"your prompt here\")",
            "clean_cache": "Manually wipe downloaded training dataset caches to free up PC space",
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
        self.last_command_type = "train"

        # Command history tracking variables
        self.command_history = []
        self.history_index = -1

        # Output collapsing state tracking
        self.group_counter = 0
        self.active_group_tag = None
        self.group_collapse_states = {}

        self.after(100, self.poll_subprocess_output)
        self.after(500, self.update_status_and_timer)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
    def create_header(self):
        header_frame = tk.Frame(self, bg="#1e1e1e", height=45)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_title = tk.Label(header_frame, text="Spider AI Image Generator UI (SPIDER CLI)", bg="#990000", fg="#ffffff", font=("Consolas", 10, "bold"), padx=10, pady=5)
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
        
        self.terminal_box.tag_config("normal", foreground="#00ff66")
        self.terminal_box.tag_config("error", foreground="#ff4444")
        self.terminal_box.tag_config("input_command", foreground="#ffffff")

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
        if not self.command_history:
            return
        if self.history_index > 0:
            self.history_index -= 1
        elif self.history_index == -1:
            self.history_index = len(self.command_history) - 1
        
        self.cmd_entry.delete(0, tk.END)
        self.cmd_entry.insert(0, self.command_history[self.history_index])
        return "break"

    def on_key_down(self, event):
        if not self.command_history:
            return
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.cmd_entry.delete(0, tk.END)
            self.cmd_entry.insert(0, self.command_history[self.history_index])
        else:
            self.history_index = len(self.command_history)
            self.cmd_entry.delete(0, tk.END)
        return "break"

    def log_input_command(self, raw_command):
        self.terminal_box.config(state=tk.NORMAL)
        self.group_counter += 1
        group_tag = f"group_{self.group_counter}"
        self.active_group_tag = group_tag
        self.group_collapse_states[group_tag] = False

        toggle_btn = tk.Button(
            self.terminal_box, 
            text="[-]", 
            bg="#1a1a1a", 
            fg="#ffcc00", 
            activebackground="#2a2a2a", 
            activeforeground="#ffcc00",
            relief=tk.FLAT, 
            font=("Consolas", 9, "bold"),
            bd=0,
            padx=2, pady=0
        )
        
        def make_toggle(gt, btn):
            return lambda: self.toggle_collapse(gt, btn)
        
        toggle_btn.config(command=make_toggle(group_tag, toggle_btn))
        self.terminal_box.window_create(tk.END, window=toggle_btn)
        
        header_text = f" > {raw_command}\n"
        self.terminal_box.insert(tk.END, header_text, "input_command")
        
        self.terminal_box.see(tk.END)
        self.terminal_box.config(state=tk.DISABLED)

    def toggle_collapse(self, group_tag, toggle_btn):
        self.terminal_box.config(state=tk.NORMAL)
        is_collapsed = self.group_collapse_states.get(group_tag, False)
        new_state = not is_collapsed
        self.group_collapse_states[group_tag] = new_state
        
        self.terminal_box.tag_config(group_tag, elide=new_state)
        
        if new_state:
            toggle_btn.config(text="[+]")
        else:
            toggle_btn.config(text="[-]")
        
        self.terminal_box.config(state=tk.DISABLED)

    def log_to_terminal(self, message, is_error=False):
        self.terminal_box.config(state=tk.NORMAL)
        color_tag = "error" if is_error else "normal"
        tags = (color_tag,)
        if self.active_group_tag:
            tags = (color_tag, self.active_group_tag)
        
        self.terminal_box.insert(tk.END, message + "\n", tags)
        self.terminal_box.see(tk.END)
        self.terminal_box.config(state=tk.DISABLED)

    def execute_user_command(self, event):
        raw_command = self.cmd_entry.get().strip()
        if not raw_command: return
        self.suggestion_listbox.pack_forget()
        self.cmd_entry.delete(0, tk.END)

        self.command_history.append(raw_command)
        self.history_index = len(self.command_history)

        if raw_command == "clear":
            self.log_input_command(raw_command)
            self.terminal_box.config(state=tk.NORMAL)
            self.terminal_box.delete("1.0", tk.END)
            self.terminal_box.config(state=tk.DISABLED)
            return
        
        if raw_command == "help":
            self.log_input_command(raw_command)
            self.log_to_terminal("Available Commands:")
            for cmd, desc in self.commands_help.items():
                self.log_to_terminal(f"  {cmd} : {desc}")
            return

        # --- MANUAL CACHE CLEANUP COMMAND ---
        if raw_command == "clean_cache":
            self.log_input_command(raw_command)
            cache_dir = "./cache_images"
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    self.log_to_terminal(f"Successfully deleted local cache directory: {cache_dir}")
                except Exception as e:
                    self.log_to_terminal(f"Error deleting cache directory: {e}", is_error=True)
            else:
                self.log_to_terminal("Cache directory './cache_images' does not exist or is already clear.")
            return
        # ------------------------------------

        # --- SECURITY CHECK & PASSWORD STRIPPING ---
        clean_command = raw_command
        if "train" in raw_command.lower():
            self.log_input_command(raw_command)
            
            expected_pass = os.environ.get("SPIDER_PASS")
            if not expected_pass:
                self.log_to_terminal("Access Denied: SPIDER_PASS environment variable is not configured.", is_error=True)
                return
                
            required_arg = f"--pass {expected_pass}"
            if required_arg not in raw_command:
                self.log_to_terminal("Access Denied: Missing or incorrect password argument in command.", is_error=True)
                return
            
            # Strip out the password argument so train_diffusion.py's argparse doesn't reject it
            clean_command = raw_command.replace(required_arg, "").strip()
        # ---------------------------------------------

        if clean_command.endswith(".py") or clean_command.startswith("train_diffusion") or clean_command.startswith("generate_image"):
            command = f"python {clean_command}"
        else:
            command = clean_command

        if "train" not in raw_command.lower():
            self.log_input_command(raw_command)

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
                    cleaned_line = line.rstrip()
                    lower_line = cleaned_line.lower()
                    is_err = any(keyword in lower_line for keyword in ["error", "exception", "traceback", "fail", "fatal"])
                    self.log_to_terminal(cleaned_line, is_error=is_err)
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
        base_cache_dir = "./cache_images"
        expected_files = [f"spe{i}.png" for i in range(1, 5)]
        self.cached_images_list = [os.path.join(base_cache_dir, f) for f in expected_files if os.path.exists(os.path.join(base_cache_dir, f))]

        if self.last_command_type == "train" and self.cached_images_list:
            cycle_interval = 5
            self.current_image_index = int(time.time() % (len(self.cached_images_list) * cycle_interval)) // cycle_interval
            if self.current_image_index < len(self.cached_images_list):
                self.load_canvas_image(self.cached_images_list[self.current_image_index])
                return

        if self.last_command_type == "generate":
            all_pngs = [f for f in glob.glob("*.png") if not f.startswith("spe")]
            if all_pngs:
                latest_gen_image = max(all_pngs, key=os.path.getmtime)
                self.load_canvas_image(latest_gen_image)
                return

        if self.cached_images_list:
            cycle_interval = 5
            self.current_image_index = int(time.time() % (len(self.cached_images_list) * cycle_interval)) // cycle_interval
            if self.current_image_index < len(self.cached_images_list):
                self.load_canvas_image(self.cached_images_list[self.current_image_index])

    def load_canvas_image(self, path):
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