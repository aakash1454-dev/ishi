#!/usr/bin/env python3
"""
ISHI Conjunctiva Cropping Tool - Crescent Edition
==================================================
A tool for manually cropping conjunctiva images using a 4-point crescent guide.

HOW TO USE:
1. Run: python crop_tool.py
2. Select input folder (raw images) and output folder (where crops go)
3. Drag the 4 control points (L, R, T, B) to fit the crescent around the conjunctiva
4. Drag the crescent body to move it
5. Press A = save as Anemic, N = save as Not Anemic, S = Skip
6. Tool auto-advances to next image

REQUIREMENTS:
- Python 3.7+
- Pillow: pip install Pillow

That's it! No other dependencies needed.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import pandas as pd
import os
import json
import math
from pathlib import Path


def bezier_point(t, p0, p1, p2):
    """Calculate point on quadratic Bezier curve at parameter t"""
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return (x, y)


def get_crescent_bbox(left, right, top_ctrl, bot_ctrl, padding=10):
    """Get bounding box of crescent shape with padding"""
    # Sample points along both curves
    points = []
    for t in [i/20 for i in range(21)]:
        points.append(bezier_point(t, left, top_ctrl, right))
        points.append(bezier_point(t, right, bot_ctrl, left))
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    return (
        min(xs) - padding,
        min(ys) - padding,
        max(xs) + padding,
        max(ys) + padding
    )


class CropTool:
    def __init__(self, root):
        self.root = root
        self.root.title("ISHI Conjunctiva Cropper - Crescent Edition")
        self.root.geometry("1300x900")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.images = []
        self.current_idx = 0
        self.input_folder = ""
        self.output_folder = ""
        self.current_image = None
        self.photo = None
        self.scale = 1.0
        self.progress_file = None
        self.completed = set()
        self.history = []  # Stack of (filename, label, filepath) for undo
        self.labels_df = None  # DataFrame with image labels
        self.load_labels()
        
        # Image position on canvas
        self.img_x = 0
        self.img_y = 0
        self.img_display_w = 0
        self.img_display_h = 0
        
        # Crescent control points (canvas coordinates)
        # L=left, R=right, T=top curve control, B=bottom curve control
        self.points = {
            'L': [200, 300],
            'R': [400, 300],
            'T': [300, 260],  # Top curve control (curves upward)
            'B': [300, 340],  # Bottom curve control (curves downward)
        }
        self.point_radius = 12
        self.dragging = None
        self.drag_offset = (0, 0)
        self.crescent_ids = []
        self.point_ids = {}
        
        self.setup_ui()
        self.bind_keys()
    
    def load_labels(self):
        """Load labels from CSV file"""
        # Try common locations for labels.csv
        possible_paths = [
            r"C:\Desktop\ironstrong-health\ishi\datasets\anemia\processed\eyes_defy_anemia\labels.csv",
            "datasets/anemia/processed/eyes_defy_anemia/labels.csv",
            "labels.csv",
        ]
        
        for path in possible_paths:
            try:
                self.labels_df = pd.read_csv(path)
                print(f"Loaded labels from {path}")
                # Create lookup by filename
                self.labels_df['filename'] = self.labels_df['image_path'].apply(lambda x: os.path.basename(x))
                return
            except:
                continue
        
        print("Warning: Could not load labels.csv - labels won't be displayed")
    
    def get_label_for_image(self, img_path):
        """Get anemia label for an image"""
        if self.labels_df is None:
            return None, None
        
        filename = os.path.basename(img_path)
        
        # Try exact match
        matches = self.labels_df[self.labels_df['filename'] == filename]
        
        if len(matches) == 0:
            # Try matching by subject folder
            parts = img_path.replace('\\', '/').split('/')
            for i, part in enumerate(parts):
                if part.isdigit():
                    subject_id = int(part)
                    matches = self.labels_df[self.labels_df['subject_id'] == subject_id]
                    break
        
        if len(matches) > 0:
            row = matches.iloc[0]
            label = int(row['label_final']) if pd.notna(row['label_final']) else None
            hb = row['hb'] if pd.notna(row['hb']) else None
            return label, hb
        
        return None, None
    
    def setup_ui(self):
        # Top frame - controls
        top_frame = tk.Frame(self.root, bg='#1a1a2e', pady=10)
        top_frame.pack(fill='x')
        
        btn_style = {'bg': '#4a4e69', 'fg': 'white', 'font': ('Arial', 11), 
                     'relief': 'flat', 'padx': 15, 'pady': 8, 'cursor': 'hand2'}
        
        tk.Button(top_frame, text="📁 Select Input Folder", command=self.select_input,
                  **btn_style).pack(side='left', padx=10)
        tk.Button(top_frame, text="📂 Select Output Folder", command=self.select_output,
                  **btn_style).pack(side='left', padx=10)
        tk.Button(top_frame, text="🔄 Reset Shape", command=self.reset_crescent,
                  bg='#e07a5f', fg='white', font=('Arial', 11), relief='flat', 
                  padx=15, pady=8, cursor='hand2').pack(side='left', padx=10)
        
        tk.Button(top_frame, text="↩️ Undo Last", command=self.undo_last,
                  bg='#9b5de5', fg='white', font=('Arial', 11), relief='flat', 
                  padx=15, pady=8, cursor='hand2').pack(side='left', padx=10)
        
        self.status_label = tk.Label(top_frame, text="Select folders to begin", 
                                      bg='#1a1a2e', fg='#9a8c98', font=('Arial', 11))
        self.status_label.pack(side='left', padx=20)
        
        # Progress
        self.progress_label = tk.Label(top_frame, text="", bg='#1a1a2e', fg='#c9ada7', 
                                        font=('Arial', 11, 'bold'))
        self.progress_label.pack(side='right', padx=20)
        
        # Canvas for image
        canvas_frame = tk.Frame(self.root, bg='#0f0f23')
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#0f0f23', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        # Bottom frame - instructions
        bottom_frame = tk.Frame(self.root, bg='#1a1a2e', pady=15)
        bottom_frame.pack(fill='x')
        
        instructions = tk.Label(bottom_frame, 
            text="🖱️ Drag L/R/T/B points to shape  |  Drag body to move  |  A = Anemic  |  N = Not Anemic  |  U = Undo  |  S = Skip  |  R = Reset  |  ←/→ = Nav",
            bg='#1a1a2e', fg='#f2e9e4', font=('Arial', 11))
        instructions.pack()
        
        # Filename display
        self.filename_label = tk.Label(bottom_frame, text="", bg='#1a1a2e', fg='#c9ada7', 
                                        font=('Arial', 10))
        self.filename_label.pack(pady=(5, 0))
        
        # Label display (shows ANEMIC / NOT ANEMIC)
        self.label_display = tk.Label(bottom_frame, text="", bg='#1a1a2e', 
                                       font=('Arial', 16, 'bold'))
        self.label_display.pack(pady=(5, 0))
    
    def bind_keys(self):
        self.root.bind('a', lambda e: self.save_crop('anemic'))
        self.root.bind('A', lambda e: self.save_crop('anemic'))
        self.root.bind('n', lambda e: self.save_crop('nonanemic'))
        self.root.bind('N', lambda e: self.save_crop('nonanemic'))
        self.root.bind('s', lambda e: self.skip_image())
        self.root.bind('S', lambda e: self.skip_image())
        self.root.bind('r', lambda e: self.reset_crescent())
        self.root.bind('R', lambda e: self.reset_crescent())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.next_image())
        self.root.bind('<Control-z>', lambda e: self.undo_last())
        self.root.bind('u', lambda e: self.undo_last())
        self.root.bind('U', lambda e: self.undo_last())
    
    def on_canvas_resize(self, event):
        if self.current_image:
            self.root.after(100, self.show_current_image)
    
    def select_input(self):
        folder = filedialog.askdirectory(title="Select folder with raw eye images")
        if folder:
            self.input_folder = folder
            self.load_images()
    
    def select_output(self):
        folder = filedialog.askdirectory(title="Select output folder for crops")
        if folder:
            self.output_folder = folder
            # Create subfolders
            os.makedirs(os.path.join(folder, 'anemic'), exist_ok=True)
            os.makedirs(os.path.join(folder, 'nonanemic'), exist_ok=True)
            self.status_label.config(text=f"Output: {Path(folder).name}/")
            
            # Load progress file
            self.progress_file = os.path.join(folder, '.crop_progress.json')
            self.load_progress()
    
    def load_progress(self):
        """Load list of already-cropped files"""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed = set(data.get('completed', []))
            except:
                self.completed = set()
        self.update_progress_display()
    
    def save_progress(self):
        """Save progress to file"""
        if self.progress_file:
            with open(self.progress_file, 'w') as f:
                json.dump({'completed': list(self.completed)}, f)
    
    def load_images(self):
        """Load all images from input folder"""
        self.images = []
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        
        # Only load .jpg files - the .png files in this dataset seem corrupted
        # You can change this if needed
        valid_extensions = {'.jpg', '.jpeg'}
        
        for root_dir, dirs, files in os.walk(self.input_folder):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in valid_extensions:
                    self.images.append(os.path.join(root_dir, f))
        
        self.images.sort()
        self.current_idx = 0
        
        # Skip to first uncompleted image
        self.skip_to_uncompleted()
        
        self.status_label.config(text=f"Loaded {len(self.images)} images from {Path(self.input_folder).name}/")
        self.update_progress_display()
        
        if self.images:
            self.show_current_image()
    
    def skip_to_uncompleted(self):
        """Skip to first image not yet cropped"""
        for i, img_path in enumerate(self.images):
            if os.path.basename(img_path) not in self.completed:
                self.current_idx = i
                return
        # All done
        self.current_idx = 0
    
    def update_progress_display(self):
        if self.images:
            done = len(self.completed)
            total = len(self.images)
            pct = (done / total * 100) if total > 0 else 0
            self.progress_label.config(text=f"Progress: {done}/{total} ({pct:.1f}%)")
    
    def reset_crescent(self):
        """Reset crescent to default position centered on image"""
        canvas_w = self.canvas.winfo_width() or 1000
        canvas_h = self.canvas.winfo_height() or 600
        
        # Default crescent size - VERTICAL orientation for portrait photos
        width = 80   # narrow width
        height = 200  # tall height
        
        cx = canvas_w // 2
        cy = canvas_h // 2
        
        # Vertical crescent: T and B are the tips, L and R control the curves
        self.points = {
            'T': [cx, cy - height//2],      # Top tip
            'B': [cx, cy + height//2],      # Bottom tip
            'L': [cx - width//2, cy],       # Left curve control
            'R': [cx + width//2, cy],       # Right curve control
        }
        self.draw_crescent()
    
    def show_current_image(self):
        if not self.images or self.current_idx >= len(self.images):
            return
        
        img_path = self.images[self.current_idx]
        
        # Try to open image, skip if corrupted
        try:
            self.current_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            # Mark as completed so we don't try again
            self.completed.add(os.path.basename(img_path))
            self.save_progress()
            # Try next image
            self.current_idx = (self.current_idx + 1) % len(self.images)
            self.show_current_image()
            return
        
        # Fit image to canvas
        canvas_w = self.canvas.winfo_width() or 1000
        canvas_h = self.canvas.winfo_height() or 600
        
        img_w, img_h = self.current_image.size
        
        # Calculate scale to fit
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale = min(scale_w, scale_h, 1.5)  # Allow slight upscale for small images
        
        self.img_display_w = int(img_w * self.scale)
        self.img_display_h = int(img_h * self.scale)
        
        display_img = self.current_image.resize((self.img_display_w, self.img_display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        
        # Center on canvas
        self.img_x = (canvas_w - self.img_display_w) // 2
        self.img_y = (canvas_h - self.img_display_h) // 2
        
        self.canvas.delete('all')
        self.canvas.create_image(self.img_x, self.img_y, anchor='nw', image=self.photo, tags='image')
        
        # Show filename
        basename = os.path.basename(img_path)
        status = "✓ Done" if basename in self.completed else ""
        self.filename_label.config(text=f"{basename}  {status}")
        
        # Show label (ANEMIC / NOT ANEMIC)
        label, hb = self.get_label_for_image(img_path)
        if label is not None:
            if label == 1:
                label_text = f"🔴 ANEMIC (Hb: {hb:.1f})" if hb else "🔴 ANEMIC"
                self.label_display.config(text=label_text, fg='#ff6b6b')
            else:
                label_text = f"🟢 NOT ANEMIC (Hb: {hb:.1f})" if hb else "🟢 NOT ANEMIC"
                self.label_display.config(text=label_text, fg='#4ecdc4')
        else:
            self.label_display.config(text="⚪ Label unknown", fg='#888888')
        
        # Reset crescent to center on image
        self.reset_crescent()
    
    def draw_crescent(self):
        """Draw the crescent shape and control points"""
        # Clear old drawings
        for cid in self.crescent_ids:
            self.canvas.delete(cid)
        self.crescent_ids = []
        for pid in self.point_ids.values():
            self.canvas.delete(pid)
        self.point_ids = {}
        
        L = self.points['L']
        R = self.points['R']
        T = self.points['T']
        B = self.points['B']
        
        # Draw crescent using line segments approximating Bezier curves
        # VERTICAL orientation: T and B are tips, L and R are curve controls
        # Left curve: T -> L -> B
        left_curve = []
        for i in range(21):
            t = i / 20
            pt = bezier_point(t, T, L, B)
            left_curve.append(pt)
        
        # Right curve: B -> R -> T
        right_curve = []
        for i in range(21):
            t = i / 20
            pt = bezier_point(t, B, R, T)
            right_curve.append(pt)
        
        # Rename for compatibility with rest of code
        top_curve = left_curve
        bot_curve = right_curve
        
        # Draw filled crescent (semi-transparent effect via stipple)
        all_points = top_curve + bot_curve
        flat_points = [coord for pt in all_points for coord in pt]
        
        if len(flat_points) >= 6:
            fill_id = self.canvas.create_polygon(
                flat_points,
                fill='#00ff88',
                stipple='gray25',
                outline='',
                tags='crescent_fill'
            )
            self.crescent_ids.append(fill_id)
        
        # Draw outline
        for i in range(len(top_curve) - 1):
            lid = self.canvas.create_line(
                top_curve[i][0], top_curve[i][1],
                top_curve[i+1][0], top_curve[i+1][1],
                fill='#00ff88', width=3, tags='crescent_line'
            )
            self.crescent_ids.append(lid)
        
        for i in range(len(bot_curve) - 1):
            lid = self.canvas.create_line(
                bot_curve[i][0], bot_curve[i][1],
                bot_curve[i+1][0], bot_curve[i+1][1],
                fill='#00ff88', width=3, tags='crescent_line'
            )
            self.crescent_ids.append(lid)
        
        # Draw control points
        point_colors = {'L': '#ff6b6b', 'R': '#ff6b6b', 'T': '#4ecdc4', 'B': '#4ecdc4'}
        for name, pos in self.points.items():
            x, y = pos
            r = self.point_radius
            
            # Outer circle (larger hit area)
            oid = self.canvas.create_oval(
                x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                fill='', outline='', tags=f'point_{name}_hit'
            )
            self.crescent_ids.append(oid)
            
            # Visible circle
            pid = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill='white', outline=point_colors[name], width=3,
                tags=f'point_{name}'
            )
            self.point_ids[name] = pid
            self.crescent_ids.append(pid)
            
            # Label
            tid = self.canvas.create_text(
                x, y, text=name, fill=point_colors[name],
                font=('Arial', 10, 'bold'), tags=f'label_{name}'
            )
            self.crescent_ids.append(tid)
    
    def get_point_at(self, x, y):
        """Check if (x, y) is near a control point"""
        for name, pos in self.points.items():
            px, py = pos
            dist = math.sqrt((x - px)**2 + (y - py)**2)
            if dist <= self.point_radius + 10:  # Extra tolerance
                return name
        return None
    
    def is_inside_crescent(self, x, y):
        """Check if point is roughly inside the crescent bounding box"""
        bbox = get_crescent_bbox(
            self.points['L'], self.points['R'],
            self.points['T'], self.points['B'],
            padding=20
        )
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    
    def on_mouse_down(self, event):
        x, y = event.x, event.y
        
        # Check if clicking on a control point
        point = self.get_point_at(x, y)
        if point:
            self.dragging = ('point', point)
            return
        
        # Check if clicking inside crescent (to drag whole thing)
        if self.is_inside_crescent(x, y):
            self.dragging = ('crescent', None)
            self.drag_offset = (x, y)
            return
        
        self.dragging = None
    
    def on_mouse_drag(self, event):
        if not self.dragging:
            return
        
        x, y = event.x, event.y
        
        if self.dragging[0] == 'point':
            # Move single point
            point_name = self.dragging[1]
            self.points[point_name] = [x, y]
            self.draw_crescent()
        
        elif self.dragging[0] == 'crescent':
            # Move entire crescent
            dx = x - self.drag_offset[0]
            dy = y - self.drag_offset[1]
            
            for name in self.points:
                self.points[name][0] += dx
                self.points[name][1] += dy
            
            self.drag_offset = (x, y)
            self.draw_crescent()
    
    def on_mouse_up(self, event):
        self.dragging = None
    
    def save_crop(self, label):
        if not self.output_folder:
            messagebox.showwarning("No Output", "Please select an output folder first!")
            return
        
        if not self.current_image:
            return
        
        # Get bounding box of crescent in canvas coordinates
        bbox = get_crescent_bbox(
            self.points['L'], self.points['R'],
            self.points['T'], self.points['B'],
            padding=5  # Smaller padding for tighter crop
        )
        
        # Convert canvas coords to image coords
        x0 = (bbox[0] - self.img_x) / self.scale
        y0 = (bbox[1] - self.img_y) / self.scale
        x1 = (bbox[2] - self.img_x) / self.scale
        y1 = (bbox[3] - self.img_y) / self.scale
        
        # Clamp to image bounds
        img_w, img_h = self.current_image.size
        x0 = max(0, min(x0, img_w))
        y0 = max(0, min(y0, img_h))
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        
        # Ensure valid crop
        if x1 <= x0 or y1 <= y0:
            messagebox.showwarning("Invalid Crop", "Crop area is too small or outside image!")
            return
        
        # Convert control points to image coordinates (relative to crop area)
        def canvas_to_crop(pt):
            ix = (pt[0] - self.img_x) / self.scale - x0
            iy = (pt[1] - self.img_y) / self.scale - y0
            return (ix, iy)
        
        L_img = canvas_to_crop(self.points['L'])
        R_img = canvas_to_crop(self.points['R'])
        T_img = canvas_to_crop(self.points['T'])
        B_img = canvas_to_crop(self.points['B'])
        
        # Crop the rectangular region first
        cropped = self.current_image.crop((int(x0), int(y0), int(x1), int(y1)))
        crop_w, crop_h = cropped.size
        
        # Create crescent mask
        mask = Image.new('L', (crop_w, crop_h), 0)  # Black mask
        mask_draw = ImageDraw.Draw(mask)
        
        # Generate crescent polygon points - VERTICAL orientation
        crescent_points = []
        # Left curve: T -> L -> B
        for i in range(21):
            t = i / 20
            pt = bezier_point(t, T_img, L_img, B_img)
            crescent_points.append(pt)
        # Right curve: B -> R -> T
        for i in range(21):
            t = i / 20
            pt = bezier_point(t, B_img, R_img, T_img)
            crescent_points.append(pt)
        
        # Draw filled crescent on mask (white = keep)
        if len(crescent_points) >= 3:
            mask_draw.polygon(crescent_points, fill=255)
        
        # Apply slight blur to mask edges for smoother result
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Create black background image
        result = Image.new('RGB', (crop_w, crop_h), (0, 0, 0))
        
        # Composite: cropped image where mask is white, black elsewhere
        result.paste(cropped, mask=mask)
        cropped = result
        
        # Save
        basename = os.path.basename(self.images[self.current_idx])
        name, ext = os.path.splitext(basename)
        # Always save as PNG for quality
        out_path = os.path.join(self.output_folder, label, f"{name}.png")
        
        cropped.save(out_path)
        
        # Mark as completed
        self.completed.add(basename)
        self.save_progress()
        self.update_progress_display()
        
        # Add to history for undo
        self.history.append({
            'basename': basename,
            'label': label,
            'filepath': out_path,
            'prev_idx': self.current_idx
        })
        
        # Flash feedback
        self.flash_feedback(label)
        
        # Next image
        self.root.after(250, self.next_uncompleted)
    
    def flash_feedback(self, label):
        color = '#ff6b6b' if label == 'anemic' else '#4ecdc4'
        text = f"✓ Saved as {label.upper()}"
        
        # Create flash overlay
        flash = tk.Label(self.root, text=text, bg=color, fg='white', 
                         font=('Arial', 18, 'bold'), padx=40, pady=20)
        flash.place(relx=0.5, rely=0.5, anchor='center')
        self.root.after(350, flash.destroy)
    
    def undo_last(self):
        """Undo the last crop - delete file and go back to that image"""
        if not self.history:
            messagebox.showinfo("Undo", "Nothing to undo!")
            return
        
        last = self.history.pop()
        
        # Delete the saved file
        try:
            if os.path.exists(last['filepath']):
                os.remove(last['filepath'])
        except Exception as e:
            print(f"Could not delete {last['filepath']}: {e}")
        
        # Remove from completed set
        self.completed.discard(last['basename'])
        self.save_progress()
        self.update_progress_display()
        
        # Go back to that image
        self.current_idx = last['prev_idx']
        self.show_current_image()
        
        # Flash feedback
        flash = tk.Label(self.root, text=f"↩️ Undid {last['basename']}", 
                         bg='#9b5de5', fg='white', 
                         font=('Arial', 14, 'bold'), padx=30, pady=15)
        flash.place(relx=0.5, rely=0.5, anchor='center')
        self.root.after(400, flash.destroy)
    
    def skip_image(self):
        self.next_uncompleted()
    
    def next_image(self):
        if self.images:
            self.current_idx = (self.current_idx + 1) % len(self.images)
            self.show_current_image()
    
    def prev_image(self):
        if self.images:
            self.current_idx = (self.current_idx - 1) % len(self.images)
            self.show_current_image()
    
    def next_uncompleted(self):
        """Go to next uncompleted image"""
        if not self.images:
            return
        
        start = self.current_idx
        for i in range(len(self.images)):
            idx = (start + 1 + i) % len(self.images)
            if os.path.basename(self.images[idx]) not in self.completed:
                self.current_idx = idx
                self.show_current_image()
                return
        
        # All done!
        messagebox.showinfo("Complete!", f"🎉 All {len(self.images)} images have been cropped!")


def main():
    root = tk.Tk()
    app = CropTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
