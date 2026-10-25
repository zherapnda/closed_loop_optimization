import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from bo_session import BOSession


class BOApp:
    def __init__(self, root):
        self.root = root
        root.title("Closed-Loop Duty Cycle Optimizer")
        root.geometry("760x560")

        self.session = BOSession()
        self.current_image_path = None

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=10)
        self.status_var = tk.StringVar(value=self.session.status_line())
        tk.Label(top, textvariable=self.status_var, anchor="w", justify="left").pack(fill="x")

        rec_frame = tk.LabelFrame(root, text="Recommended experiment", padx=10, pady=10)
        rec_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.rec_var = tk.StringVar(value="")
        tk.Label(rec_frame, textvariable=self.rec_var, font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x")

        id_row = tk.Frame(rec_frame)
        id_row.pack(fill="x", pady=(8, 0))
        tk.Label(id_row, text="Physical structure ID:").pack(side="left")
        self.structure_id_var = tk.StringVar(value="")
        tk.Entry(id_row, textvariable=self.structure_id_var, width=20).pack(side="left", padx=8)

        img_row = tk.Frame(rec_frame)
        img_row.pack(fill="x", pady=(8, 0))
        tk.Button(img_row, text="Select Microscope Image...", command=self.select_image).pack(side="left")
        self.image_var = tk.StringVar(value="No image selected")
        tk.Label(img_row, textvariable=self.image_var, anchor="w").pack(side="left", padx=8)

        self.submit_btn = tk.Button(rec_frame, text="Compute && Submit", command=self.submit, state="disabled")
        self.submit_btn.pack(anchor="w", pady=(8, 0))

        self.result_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.result_var, anchor="w", justify="left", fg="#0a5").pack(
            fill="x", padx=10, pady=(0, 10))

        columns = ("exp", "amplitude", "rise", "dwell", "fall", "duty_cycle", "std", "output")
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=10)
        headings = {"exp": "#", "amplitude": "Amplitude", "rise": "Rise", "dwell": "Dwell", "fall": "Fall",
                    "duty_cycle": "Duty Cycle (%)", "std": "Std (%)", "output": "Output"}
        widths = {"exp": 40, "amplitude": 90, "rise": 70, "dwell": 70, "fall": 70,
                  "duty_cycle": 100, "std": 80, "output": 90}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        self.best_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self.best_var, anchor="w", justify="left").pack(fill="x")

        self.refresh_recommendation()

    def refresh_recommendation(self):
        done, reason = self.session.is_finished()
        if done:
            self.rec_var.set(f"Finished: {reason}")
            self.submit_btn.config(state="disabled")
            self.show_best()
            return

        params = self.session.next_params()
        n = self.session.experiment_num + 1
        source = "Latin Hypercube (initial exploration)" if self.session.pending_source == "lhs" \
            else "GP + Expected Improvement"
        self.rec_var.set(
            f"Experiment {n}  [{source}]\n"
            f"Amplitude: {params[0]:.2f}   Rise: {params[1]:.2f}   Dwell: {params[2]:.2f}   Fall: {params[3]:.2f}"
        )
        self.status_var.set(self.session.status_line())

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select microscope image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.current_image_path = path
            self.image_var.set(path)
            self.submit_btn.config(state="normal")

    def submit(self):
        if not self.current_image_path:
            messagebox.showinfo("No image", "Select a microscope image first.")
            return
        structure_id = self.structure_id_var.get().strip()
        if not structure_id:
            messagebox.showinfo("No structure ID", "Enter the physical structure ID first.")
            return

        self.submit_btn.config(state="disabled")
        self.result_var.set("Computing duty cycle...")
        self.root.update()

        try:
            result = self.session.submit_result(structure_id, self.current_image_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.submit_btn.config(state="normal")
            return

        if result is None:
            messagebox.showwarning(
                "No numeric duty cycle",
                "Every column failed or landed in a problem segment for that image -- "
                "select a different image for this same recommendation.",
            )
            self.submit_btn.config(state="normal")
            return

        mean, std, output, diagnostics = result
        warning = ""
        if diagnostics["classification"] == "spatially_varying":
            warning = (f"  [spatially varying: {diagnostics['problem_fraction']:.0%} of ROI overpoled/unstable, "
                        f"penalty included in output]")
        self.result_var.set(f"Duty cycle: {mean:.2f}% ± {std:.2f}%   Output: {output:.4f}{warning}")

        params = self.session.X_real[-1].tolist()
        self.tree.insert("", "end", values=(
            self.session.experiment_num, f"{params[0]:.2f}", f"{params[1]:.2f}", f"{params[2]:.2f}",
            f"{params[3]:.2f}", f"{mean:.2f}", f"{std:.2f}", f"{output:.4f}",
        ))

        self.current_image_path = None
        self.image_var.set("No image selected")
        self.structure_id_var.set("")
        self.refresh_recommendation()

    def show_best(self):
        best = self.session.best_result()
        if best is None:
            self.best_var.set("No real experiments were run.")
            return
        params, output = best
        self.best_var.set(
            f"BEST FOUND -- Amplitude: {params[0]:.2f}, Rise: {params[1]:.2f}, "
            f"Dwell: {params[2]:.2f}, Fall: {params[3]:.2f}  |  Output: {output:.4f}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    BOApp(root)
    root.mainloop()
