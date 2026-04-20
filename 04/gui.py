from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = SCRIPT_DIR / "main.py"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


class RenderGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Lab 04 Path Tracer")
        self.root.geometry("980x720")
        self.root.minsize(880, 620)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.vars = {
            "width": tk.StringVar(value="500"),
            "height": tk.StringVar(value="500"),
            "samples": tk.StringVar(value="32"),
            "max_depth": tk.StringVar(value="5"),
            "rr_depth": tk.StringVar(value="3"),
            "white_point": tk.StringVar(value="3.5"),
            "workers": tk.StringVar(value="4"),
            "chunk_rows": tk.StringVar(value="16"),
            "scene": tk.StringVar(value="cornell"),
            "material_mode": tk.StringVar(value="balanced"),
            "camera": tk.StringVar(value="0 1.05 3.35"),
            "look_at": tk.StringVar(value="0 0.92 0"),
            "fov": tk.StringVar(value="42"),
            "light_scale": tk.StringVar(value="1.0"),
            "obj": tk.StringVar(value=""),
            "obj_scale": tk.StringVar(value="0.45"),
            "obj_offset": tk.StringVar(value="0 0 0.15"),
            "output": tk.StringVar(value=str(OUTPUT_DIR / "gui_render.ppm")),
        }
        self.status = tk.StringVar(value="Ready")

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_log_queue)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        controls = ttk.Frame(root_frame)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        log_frame = ttk.Frame(root_frame)
        log_frame.grid(row=0, column=1, sticky="nsew")
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._build_image_section(controls)
        self._build_scene_section(controls)
        self._build_camera_section(controls)
        self._build_output_section(controls)
        self._build_buttons(controls)

        ttk.Label(log_frame, textvariable=self.status).grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(log_frame, height=24, wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.progress = ttk.Progressbar(log_frame, mode="indeterminate")
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_image_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Image and quality", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        self._entry(frame, "Width", "width", 0, 0)
        self._entry(frame, "Height", "height", 0, 2)
        self._entry(frame, "Samples", "samples", 1, 0)
        self._entry(frame, "Max depth", "max_depth", 1, 2)
        self._entry(frame, "RR depth", "rr_depth", 2, 0)
        self._entry(frame, "Workers", "workers", 2, 2)
        self._entry(frame, "Chunk rows", "chunk_rows", 3, 0)
        self._entry(frame, "White point", "white_point", 3, 2)

    def _build_scene_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Scene and material", padding=10)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Label(frame, text="Scene").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.vars["scene"],
            values=("cornell", "mirror-test"),
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)

        ttk.Label(frame, text="Material").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.vars["material_mode"],
            values=("balanced", "diffuse", "mirror"),
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)

        self._entry(frame, "Light scale", "light_scale", 2, 0)

        ttk.Label(frame, text="OBJ").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.vars["obj"], width=28).grid(row=3, column=1, sticky="ew", padx=(8, 4), pady=3)
        ttk.Button(frame, text="Browse", command=self._browse_obj).grid(row=3, column=2, sticky="ew", pady=3)

        self._entry(frame, "OBJ scale", "obj_scale", 4, 0)
        self._entry(frame, "OBJ offset", "obj_offset", 5, 0, columnspan=2)
        frame.columnconfigure(1, weight=1)

    def _build_camera_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Camera", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        self._entry(frame, "Camera xyz", "camera", 0, 0, columnspan=2)
        self._entry(frame, "Look at xyz", "look_at", 1, 0, columnspan=2)
        self._entry(frame, "FOV", "fov", 2, 0)

    def _build_output_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Output", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        ttk.Label(frame, text="PPM path").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.vars["output"], width=34).grid(row=0, column=1, sticky="ew", padx=(8, 4), pady=3)
        ttk.Button(frame, text="Browse", command=self._browse_output).grid(row=0, column=2, sticky="ew", pady=3)
        frame.columnconfigure(1, weight=1)

    def _build_buttons(self, parent: ttk.Frame) -> None:
        presets = ttk.LabelFrame(parent, text="Presets", padding=10)
        presets.pack(fill="x", pady=(0, 10))
        ttk.Button(presets, text="Preview", command=lambda: self._apply_preset("preview")).pack(side="left", fill="x", expand=True)
        ttk.Button(presets, text="Balanced", command=lambda: self._apply_preset("balanced")).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(presets, text="Final", command=lambda: self._apply_preset("final")).pack(side="left", fill="x", expand=True, padx=(8, 0))

        frame = ttk.Frame(parent)
        frame.pack(fill="x")
        self.start_button = ttk.Button(frame, text="Start render", command=self._start_render)
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(frame, text="Stop", command=self._stop_render, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(frame, text="Open PNG", command=self._open_png).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(frame, text="Open folder", command=self._open_output_folder).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _entry(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=self.vars[key], width=18)
        entry.grid(row=row, column=column + 1, columnspan=columnspan, sticky="ew", padx=(8, 10), pady=3)
        parent.columnconfigure(column + 1, weight=1)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=OUTPUT_DIR,
            defaultextension=".ppm",
            filetypes=(("PPM image", "*.ppm"), ("All files", "*.*")),
        )
        if path:
            self.vars["output"].set(path)

    def _browse_obj(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=SCRIPT_DIR,
            filetypes=(("OBJ mesh", "*.obj"), ("All files", "*.*")),
        )
        if path:
            self.vars["obj"].set(path)

    def _build_command(self) -> list[str]:
        output_path = Path(self.vars["output"].get().strip() or OUTPUT_DIR / "gui_render.ppm")
        if output_path.suffix.lower() != ".ppm":
            output_path = output_path.with_suffix(".ppm")
            self.vars["output"].set(str(output_path))

        command = [
            sys.executable,
            "-u",
            str(MAIN_SCRIPT),
            "--width",
            self._int_value("width"),
            "--height",
            self._int_value("height"),
            "--samples",
            self._int_value("samples"),
            "--max-depth",
            self._int_value("max_depth"),
            "--rr-depth",
            self._int_value("rr_depth"),
            "--workers",
            self._int_value("workers"),
            "--chunk-rows",
            self._int_value("chunk_rows"),
            "--scene",
            self.vars["scene"].get(),
            "--material-mode",
            self.vars["material_mode"].get(),
            "--camera",
            *self._vec_values("camera"),
            "--look-at",
            *self._vec_values("look_at"),
            "--fov",
            self._float_value("fov"),
            "--light-scale",
            self._float_value("light_scale"),
            "--output",
            str(output_path),
        ]

        white_point = self.vars["white_point"].get().strip()
        if white_point:
            command.extend(["--white-point", self._float_value("white_point")])

        obj_path = self.vars["obj"].get().strip()
        if obj_path:
            command.extend(
                [
                    "--obj",
                    obj_path,
                    "--obj-scale",
                    self._float_value("obj_scale"),
                    "--obj-offset",
                    *self._vec_values("obj_offset"),
                ]
            )
        return command

    def _apply_preset(self, preset: str) -> None:
        if preset == "preview":
            values = {
                "width": "240",
                "height": "240",
                "samples": "2",
                "max_depth": "3",
                "rr_depth": "3",
                "workers": "4",
                "chunk_rows": "24",
                "output": str(OUTPUT_DIR / "preview.ppm"),
            }
        elif preset == "balanced":
            values = {
                "width": "500",
                "height": "500",
                "samples": "16",
                "max_depth": "4",
                "rr_depth": "3",
                "workers": "4",
                "chunk_rows": "16",
                "output": str(OUTPUT_DIR / "balanced.ppm"),
            }
        else:
            values = {
                "width": "500",
                "height": "500",
                "samples": "64",
                "max_depth": "5",
                "rr_depth": "3",
                "workers": "4",
                "chunk_rows": "16",
                "output": str(OUTPUT_DIR / "final.ppm"),
            }

        for key, value in values.items():
            self.vars[key].set(value)

    def _int_value(self, key: str) -> str:
        value = int(self.vars[key].get().strip())
        if value <= 0:
            raise ValueError(f"{key} must be positive.")
        return str(value)

    def _float_value(self, key: str) -> str:
        return str(float(self.vars[key].get().strip().replace(",", ".")))

    def _vec_values(self, key: str) -> list[str]:
        values = self.vars[key].get().replace(",", ".").split()
        if len(values) != 3:
            raise ValueError(f"{key} must contain exactly three numbers.")
        return [str(float(value)) for value in values]

    def _start_render(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Render is running", "Stop the current render before starting a new one.")
            return

        try:
            command = self._build_command()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error))
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.log_text.delete("1.0", tk.END)
        self._append_log("> " + " ".join(command) + "\n\n")

        try:
            self.process = subprocess.Popen(
                command,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self.process = None
            messagebox.showerror("Cannot start render", str(error))
            return

        self.status.set(f"Running, pid={self.process.pid}")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)
        return_code = self.process.wait()
        self.log_queue.put(f"\nprocess finished with code {return_code}\n")
        self.log_queue.put("__PROCESS_DONE__")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "__PROCESS_DONE__":
                    self._render_finished()
                else:
                    self._append_log(message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _render_finished(self) -> None:
        self.process = None
        self.progress.stop()
        self.status.set("Ready")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _stop_render(self) -> None:
        if self.process is None:
            return
        pid = self.process.pid
        self._append_log(f"\nstopping render tree, pid={pid}\n")
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            self.process.terminate()

    def _open_png(self) -> None:
        png_path = Path(self.vars["output"].get()).with_suffix(".png")
        if not png_path.exists():
            messagebox.showinfo("PNG not found", f"No PNG file yet:\n{png_path}")
            return
        os.startfile(png_path)

    def _open_output_folder(self) -> None:
        output_path = Path(self.vars["output"].get())
        folder = output_path.parent if output_path.parent else OUTPUT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno("Render is running", "Stop render and close the UI?"):
                return
            self._stop_render()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RenderGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
