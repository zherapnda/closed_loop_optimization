import os
import csv
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2
from PIL import Image, ImageTk

import tip_lines_check as tlc

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PREVIEW_MAX = 640  # max width/height for the preview panel
PARAM_FIELDS = ("amplitude", "rise", "dwell", "fall")


class DutyCycleApp:
    def __init__(self, root):
        self.root = root
        # one dict per image: filename, path, mean, std, problem_fraction, classification,
        # boundary_png, error, amplitude, rise, dwell, fall (params start blank -- typed in by hand)
        self.results = []
        self.tk_preview_img = None
        self._suspend_param_trace = False

        root.title("Duty Cycle Batch Calculator")
        root.geometry("1360x700")

        top_frame = tk.Frame(root)
        top_frame.pack(fill="x", padx=10, pady=10)
        self.folder_var = tk.StringVar(value="No folder selected")
        tk.Label(top_frame, textvariable=self.folder_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(top_frame, text="Select Folder...", command=self.select_folder).pack(side="right")

        self.status_var = tk.StringVar(value="Select a folder of images to begin.")
        tk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10)

        self.progress = ttk.Progressbar(root, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 10))

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        table_frame = tk.Frame(main_frame)
        table_frame.pack(side="left", fill="both", expand=True)

        columns = ("filename", "duty_cycle", "std", "problem_fraction", "classification",
                   "amplitude", "rise", "dwell", "fall")
        headings = {"filename": "Image", "duty_cycle": "Duty Cycle (%)", "std": "Std (%)",
                    "problem_fraction": "Problem Frac.", "classification": "Classification",
                    "amplitude": "Amplitude", "rise": "Rise", "dwell": "Dwell", "fall": "Fall"}
        widths = {"filename": 170, "duty_cycle": 95, "std": 75, "problem_fraction": 90,
                  "classification": 110, "amplitude": 75, "rise": 55, "dwell": 55, "fall": 55}
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="center")
        self.tree.tag_configure("problem", background="#ffe0b3")
        self.tree.tag_configure("overpoled", background="#ffb3e6")
        self.tree.tag_configure("error", background="#ffb3b3")
        self.tree.tag_configure("incomplete", background="#f0f0f0")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="left", fill="y")

        preview_frame = tk.Frame(main_frame, width=PREVIEW_MAX + 20)
        preview_frame.pack(side="left", fill="both", padx=(10, 0))
        preview_frame.pack_propagate(False)

        self.preview_label_var = tk.StringVar(value="Select a row to preview its boundary detection.")
        tk.Label(preview_frame, textvariable=self.preview_label_var, anchor="w", wraplength=PREVIEW_MAX,
                 justify="left").pack(side="top", fill="x", pady=(0, 4))

        # editable experiment-parameter panel for the currently selected row -- there's no way to
        # derive amplitude/rise/dwell/fall from the image itself, so these are typed in by hand per
        # row and committed back into that row's data on Enter/focus-out. Packed with side="bottom"
        # and BEFORE the canvas below, so it always reserves its (small, fixed) space at the bottom
        # of the fixed-height preview_frame -- previously packed after the canvas's expand=True
        # fill, which let the canvas claim all the space and push this panel off-screen.
        param_frame = tk.LabelFrame(preview_frame, text="Params (selected row)", padx=4, pady=4)
        param_frame.pack(side="bottom", fill="x", pady=(4, 0))
        self.param_vars = {}
        self.selected_index = None
        short_labels = {"amplitude": "A", "rise": "R", "dwell": "D", "fall": "F"}
        for i, field in enumerate(PARAM_FIELDS):
            tk.Label(param_frame, text=short_labels[field] + ":").grid(row=0, column=i * 2, sticky="e", padx=(2, 1), pady=2)
            var = tk.StringVar(value="")
            entry = tk.Entry(param_frame, textvariable=var, width=6)
            entry.grid(row=0, column=i * 2 + 1, sticky="w", pady=2)
            entry.bind("<Return>", self.commit_params)
            entry.bind("<FocusOut>", self.commit_params)
            self.param_vars[field] = var

        # scrollable canvas -- the boundary-lines image is upscaled and often much taller than
        # the preview pane, so the whole thing being visible at once matters less than being
        # able to scroll through it to actually check the magenta lines / green ticks. Packed
        # last so it fills whatever space is left above the param panel, never over it.
        canvas_holder = tk.Frame(preview_frame)
        canvas_holder.pack(side="top", fill="both", expand=True)
        self.preview_canvas = tk.Canvas(canvas_holder, bg="black", width=PREVIEW_MAX, height=PREVIEW_MAX - 160)
        vbar = ttk.Scrollbar(canvas_holder, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.configure(yscrollcommand=vbar.set)
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="left", fill="y")

        bottom_frame = tk.Frame(root)
        bottom_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.summary_var = tk.StringVar(value="")
        tk.Label(bottom_frame, textvariable=self.summary_var, anchor="w").pack(side="left")
        tk.Button(bottom_frame, text="Save Warm-Start CSV...", command=self.save_csv).pack(side="right")

    # ------------------------------------------------------------------ batch processing
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return
        self.folder_var.set(folder)
        self.process_folder(folder)

    def process_folder(self, folder):
        image_paths = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        self.results = []
        self.tree.delete(*self.tree.get_children())
        self.summary_var.set("")
        self.preview_canvas.delete("all")
        self.preview_label_var.set("Select a row to preview its boundary detection.")

        if not image_paths:
            self.status_var.set("No images found in that folder.")
            return

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        self.progress["maximum"] = len(image_paths)
        self.progress["value"] = 0

        for i, path in enumerate(image_paths, start=1):
            name = os.path.basename(path)
            self.status_var.set(f"Processing {i}/{len(image_paths)}: {name}")
            self.root.update()

            row = {"filename": name, "path": path, "mean": None, "std": None,
                   "problem_fraction": None, "classification": None, "boundary_png": None,
                   "error": None, "amplitude": "", "rise": "", "dwell": "", "fall": ""}
            try:
                angle = tlc.get_manual_angle(path) or 0.0
                mean, std, diagnostics = tlc.estimate_duty_cycle(path, angle=angle)
                row["mean"], row["std"] = mean, std
                row["problem_fraction"] = diagnostics["problem_fraction"]
                row["classification"] = diagnostics["classification"]
                # per-stem output, so every image in the batch keeps its own file rather than
                # overwriting a shared path -- upscale=2 (not the default 4) to keep a folder
                # of images processing at a reasonable speed; still plenty sharp to double-check
                row["boundary_png"] = tlc.draw_domain_boundary_lines(path, angle=angle, upscale=2)
            except Exception as e:
                row["error"] = str(e)

            self.results.append(row)
            self._insert_row(row)
            self.progress["value"] = i
            self.root.update()

        successes = [r for r in self.results if r["mean"] is not None]
        if successes:
            import statistics
            dcs = [r["mean"] for r in successes]
            self.summary_var.set(
                f"Done. {len(successes)}/{len(self.results)} produced a numeric duty cycle. "
                f"Avg across images: {statistics.mean(dcs):.2f}%"
                + (f" (std {statistics.stdev(dcs):.2f}%)" if len(dcs) > 1 else "")
            )
        else:
            self.summary_var.set("Done, but no images produced a numeric duty cycle -- see the table.")
        self.status_var.set(f"Finished processing {len(self.results)} image(s). "
                             f"Click a row to preview it and fill in its process parameters.")

    def _insert_row(self, row):
        if row["error"]:
            values = (row["filename"], "ERROR", row["error"], "", "", "", "", "", "")
            tags = ("error",)
        else:
            # when there's no numeric duty cycle, show WHY in the duty-cycle column itself rather
            # than a bare "n/a" -- e.g. "severely_overpoled" (domain A merged into its neighbors,
            # no boundary anywhere to measure) is itself a meaningful result, not a failure
            dc_str = f"{row['mean']:.2f}" if row["mean"] is not None else (row["classification"] or "n/a").upper()
            std_str = f"{row['std']:.2f}" if row["std"] is not None else ""
            pf_str = f"{row['problem_fraction']:.0%}" if row["problem_fraction"] is not None else ""
            values = (row["filename"], dc_str, std_str, pf_str, row["classification"] or "",
                       row["amplitude"], row["rise"], row["dwell"], row["fall"])
            if row["classification"] == "spatially_varying":
                tags = ("problem",)
            elif row["classification"] == "severely_overpoled":
                tags = ("overpoled",)
            else:
                tags = ()
        self.tree.insert("", "end", values=values, tags=tags)

    def _refresh_row_display(self, index):
        row = self.results[index]
        item_id = self.tree.get_children()[index]
        if row["error"]:
            values = (row["filename"], "ERROR", row["error"], "", "", "", "", "", "")
            tags = ("error",)
        else:
            # when there's no numeric duty cycle, show WHY in the duty-cycle column itself rather
            # than a bare "n/a" -- e.g. "severely_overpoled" (domain A merged into its neighbors,
            # no boundary anywhere to measure) is itself a meaningful result, not a failure
            dc_str = f"{row['mean']:.2f}" if row["mean"] is not None else (row["classification"] or "n/a").upper()
            std_str = f"{row['std']:.2f}" if row["std"] is not None else ""
            pf_str = f"{row['problem_fraction']:.0%}" if row["problem_fraction"] is not None else ""
            has_params = all(row[f] != "" for f in PARAM_FIELDS)
            if row["classification"] == "spatially_varying":
                tags = ("problem",)
            elif row["classification"] == "severely_overpoled":
                tags = ("overpoled",)
            else:
                tags = ()
            if row["mean"] is not None and not has_params:
                tags = tags + ("incomplete",)
            values = (row["filename"], dc_str, std_str, pf_str, row["classification"] or "",
                       row["amplitude"], row["rise"], row["dwell"], row["fall"])
        self.tree.item(item_id, values=values, tags=tags)

    # ------------------------------------------------------------------ selection / preview
    def on_select_row(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        index = self.tree.index(selection[0])
        if index >= len(self.results):
            return
        self.selected_index = index
        row = self.results[index]

        self._suspend_param_trace = True
        for field in PARAM_FIELDS:
            self.param_vars[field].set(row[field])
        self._suspend_param_trace = False

        if row["error"]:
            self.preview_canvas.delete("all")
            self.preview_label_var.set(f"{row['filename']}: ERROR -- {row['error']}")
            return

        dc_str = f"{row['mean']:.2f}%" if row["mean"] is not None else (row["classification"] or "n/a").upper()
        std_str = f"{row['std']:.2f}%" if row["std"] is not None else "n/a"
        pf_str = f"{row['problem_fraction']:.0%}" if row["problem_fraction"] is not None else "n/a"
        self.preview_label_var.set(
            f"{row['filename']}\nduty cycle: {dc_str}  Β±{std_str}   "
            f"problem_fraction: {pf_str}   classification: {row['classification']}"
        )

        self.preview_canvas.delete("all")
        path = row["boundary_png"]
        if not path or not os.path.exists(path):
            self.preview_canvas.create_text(
                PREVIEW_MAX // 2, 100, fill="white",
                text="No boundary-lines image was produced for this file.", justify="center",
            )
            return

        img = cv2.imread(path)
        if img is None:
            return
        h, w = img.shape[:2]
        # fit width to the preview pane; height is free to scroll, so the whole tall
        # boundary-lines image stays reviewable instead of squished to fit one screen
        scale = min(PREVIEW_MAX / w, 3.0)
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
        resized = cv2.resize(img, (disp_w, disp_h), interpolation=interp)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_preview_img = ImageTk.PhotoImage(Image.fromarray(resized))
        self.preview_canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.tk_preview_img)

    def commit_params(self, event):
        if self._suspend_param_trace or self.selected_index is None:
            return
        row = self.results[self.selected_index]
        for field in PARAM_FIELDS:
            val = self.param_vars[field].get().strip()
            if val:
                try:
                    float(val)
                except ValueError:
                    messagebox.showwarning("Invalid value", f"'{field}' must be numeric -- not saved.")
                    continue
            row[field] = val
        self._refresh_row_display(self.selected_index)

    # ------------------------------------------------------------------ export
    def save_csv(self):
        if not self.results:
            messagebox.showinfo("Nothing to save", "Process a folder first.")
            return
        complete = [r for r in self.results
                    if r["mean"] is not None and all(r[f] != "" for f in PARAM_FIELDS)]
        skipped = len(self.results) - len(complete)
        if not complete:
            messagebox.showwarning(
                "Nothing complete",
                "No rows have both a numeric duty cycle AND all four process parameters filled in -- "
                "nothing to export yet."
            )
            return

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        # header matches gp_trial/warm_start_data.csv exactly (amplitude,rise,dwell,fall,mean,std) --
        # build_warm_start() reads by column name, so the extra trailing columns are harmless and
        # kept here for traceability back to which image/measurement each row came from
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["amplitude", "rise", "dwell", "fall", "mean", "std",
                              "filename", "problem_fraction", "classification"])
            for r in complete:
                writer.writerow([r["amplitude"], r["rise"], r["dwell"], r["fall"], r["mean"], r["std"],
                                  r["filename"], f"{r['problem_fraction']:.4f}", r["classification"]])
        msg = f"Saved {len(complete)} row(s) to {path}"
        if skipped:
            msg += f"\n({skipped} row(s) skipped -- missing duty cycle or process parameters)"
        messagebox.showinfo("Saved", msg)


if __name__ == "__main__":
    root = tk.Tk()
    DutyCycleApp(root)
    root.mainloop()
