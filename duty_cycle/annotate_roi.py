import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import cv2
from PIL import Image, ImageTk

MAX_DISPLAY = 700  # cap the displayed image size (px) so the window can't grow taller than the screen


class AnnotateApp:
    def __init__(self, root, image_path=None):
        self.root = root
        root.title("ROI Annotator")

        top = tk.Frame(root)
        top.pack(side="top", fill="x", padx=8, pady=8)
        tk.Button(top, text="Open Image...", command=self.open_image).pack(side="left")
        self.path_var = tk.StringVar(value="No image loaded")
        tk.Label(top, textvariable=self.path_var, anchor="w").pack(side="left", padx=8)

        # packed with side="bottom" and packed BEFORE the canvas, so this bar
        # is always anchored at the bottom of the window and stays visible no
        # matter how large the image/canvas ends up -- otherwise a big enough
        # canvas can push it below the visible window with no way to reach it
        bottom = tk.Frame(root)
        bottom.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        self.status_var = tk.StringVar(value="Click and drag on the image to draw a box.")
        tk.Label(bottom, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(bottom, text="Undo Last Box", command=self.undo_last).pack(side="right", padx=(4, 0))
        tk.Button(bottom, text="Save Annotations...", command=self.save).pack(side="right", padx=(4, 0))

        # canvas lives inside a scrollable frame as a safety net -- even with
        # MAX_DISPLAY capping the zoom, this guarantees the bottom bar can
        # never be pushed off-screen by a large image
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))
        self.hbar = tk.Scrollbar(canvas_frame, orient="horizontal")
        self.vbar = tk.Scrollbar(canvas_frame, orient="vertical")
        self.canvas = tk.Canvas(canvas_frame, bg="black", cursor="crosshair",
                                 xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.hbar.config(command=self.canvas.xview)
        self.vbar.config(command=self.canvas.yview)
        self.vbar.pack(side="right", fill="y")
        self.hbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.boxes = []  # list of (label, x1, y1, x2, y2) in ORIGINAL image pixel coords
        self.cv_img = None
        self.tk_img = None
        self.rect_id = None
        self.start_xy = None
        self.image_path = None
        self.zoom = 1.0

        if image_path:
            self.load_image(image_path)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Failed to load", f"Could not read image:\n{path}")
            return
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        if norm.ndim == 2:
            norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        else:
            norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)

        self.cv_img = norm
        self.image_path = path
        self.boxes = []
        self.path_var.set(f"{os.path.basename(path)}  ({norm.shape[1]}x{norm.shape[0]}px)")
        self.status_var.set("Click and drag on the image to draw a box.")

        h, w = norm.shape[:2]
        # scale to fill MAX_DISPLAY without exceeding it in either dimension --
        # zooms small images up for easier clicking, shrinks large ones down
        self.zoom = max(0.1, min(MAX_DISPLAY / w, MAX_DISPLAY / h))
        disp_w, disp_h = max(1, int(w * self.zoom)), max(1, int(h * self.zoom))
        interp = cv2.INTER_NEAREST if self.zoom >= 1 else cv2.INTER_AREA
        display = cv2.resize(norm, (disp_w, disp_h), interpolation=interp)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(display))
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def on_press(self, event):
        if self.cv_img is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.start_xy = (x, y)
        self.rect_id = self.canvas.create_rectangle(x, y, x, y, outline="yellow", width=2)

    def on_drag(self, event):
        if self.rect_id is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x0, y0 = self.start_xy
        self.canvas.coords(self.rect_id, x0, y0, x, y)
        orig = self._to_original(x0, y0, x, y)
        self.status_var.set(f"Dragging: x={orig[0]}-{orig[2]}, y={orig[1]}-{orig[3]} (original px)")

    def on_release(self, event):
        if self.rect_id is None or self.start_xy is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x0, y0 = self.start_xy
        x1, y1, x2, y2 = self._to_original(x0, y0, x, y)
        self.start_xy = None
        self.rect_id = None

        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self.status_var.set("Box too small, ignored. Click and drag on the image to draw a box.")
            self._redraw()
            return

        label = simpledialog.askstring("Label this box", "Short label (e.g. 'overpoled', 'normal'):", parent=self.root)
        if label is None:
            self._redraw()
            return
        label = label.strip() or f"box{len(self.boxes) + 1}"
        self.boxes.append((label, x1, y1, x2, y2))
        self._redraw()
        self.status_var.set(f"Added '{label}': x={x1}-{x2}, y={y1}-{y2} (original px). {len(self.boxes)} box(es) total.")

    def _to_original(self, x0, y0, x1, y1):
        ox1, ox2 = sorted([int(x0 / self.zoom), int(x1 / self.zoom)])
        oy1, oy2 = sorted([int(y0 / self.zoom), int(y1 / self.zoom)])
        return ox1, oy1, ox2, oy2

    def _redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        colors = ["yellow", "cyan", "magenta", "lime", "orange"]
        for i, (label, x1, y1, x2, y2) in enumerate(self.boxes):
            color = colors[i % len(colors)]
            self.canvas.create_rectangle(x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom,
                                          outline=color, width=2)
            self.canvas.create_text(x1 * self.zoom + 4, y1 * self.zoom + 4, anchor="nw", text=label, fill=color,
                                     font=("Segoe UI", 10, "bold"))

    def undo_last(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw()
            self.status_var.set(f"Removed last box. {len(self.boxes)} box(es) total.")

    def save(self):
        if not self.boxes:
            messagebox.showinfo("Nothing to save", "Draw at least one box first.")
            return
        default_name = os.path.splitext(os.path.basename(self.image_path or "annotations"))[0] + "_boxes.json"
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=default_name,
                                             filetypes=[("JSON files", "*.json")])
        if not path:
            return
        data = {
            "image_path": self.image_path,
            "boxes": [{"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2} for label, x1, y1, x2, y2 in self.boxes],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(json.dumps(data, indent=2))
        messagebox.showinfo("Saved", f"Saved {len(self.boxes)} box(es) to:\n{path}\n\n(also printed to console)")


if __name__ == "__main__":
    import sys
    root = tk.Tk()
    app = AnnotateApp(root, image_path=sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()
