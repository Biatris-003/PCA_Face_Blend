import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd

USE_CLAHE = True
K_CAP = 150
TARGET_SIZE = (256, 256)  #resizing all imgs to 256*256

def preprocess_bgr(bgr, use_clahe=USE_CLAHE):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)     #BGR --> Grayscale
    gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)    #Resize
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #histogram equalization
        gray = clahe.apply(gray)
    return gray

def load_folder_vectors(folder):    #load imgs from folder and convert to vector
    items = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        bgr = cv2.imread(fpath)
        if bgr is None:
            print(f"[WARN] cannot read {fpath}, skipping.")
            continue
        gray = preprocess_bgr(bgr)
        vec = gray.flatten().astype(np.float32)
        items.append((fpath, vec))
    return items

def compute_pca(data_matrix, k):    #PCA is computed by SVD
    """
    data_matrix: (D, N) where D=dimensions, N=samples
    Returns: mean_vec (D,), eigvecs (D, k)
    """
    mean_vec = np.mean(data_matrix, axis=1, keepdims=True)
    Xc = data_matrix - mean_vec
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, U.shape[1])
    eigvecs = U[:, :k]
    return mean_vec.squeeze(), eigvecs

def project_to_pca(vec, mean_vec, eigvecs):
    return eigvecs.T @ (vec - mean_vec)

def reconstruct_from_pca(coeffs, mean_vec, eigvecs):
    return mean_vec + eigvecs @ coeffs

def vec_to_img_uint8(vec, img_shape):
    img = vec.reshape(img_shape)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) * (255.0 / (mx - mn))
    return np.clip(img, 0, 255).astype(np.uint8)

class EightFolderPCAApp:
    def __init__(self, master):
        self.master = master
        master.title("8-Folder PCA Averaging System with Face Ranking")
        master.geometry("1600x950")
        master.resizable(True, True)

        self.folders = [None] * 8
        self.folder_data = [None] * 8  
        self.global_result = None 
        self.all_global_coeffs = None 
        self.all_items_info = []  # Store (filepath, vector, folder_idx) for all images
        
        self.photos = [None] * 9  # 8 folders + 1 global
        self.random_photos = [None] * 5  # 5 random samples

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.master)
        top.pack(fill="x", padx=8, pady=3)

        tk.Label(top, text="Select 8 Folders (each with 4-8 images)", 
                font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 3))

        folder_frame = tk.Frame(top)
        folder_frame.pack(fill="x", pady=3)

        self.folder_labels = []
        for i in range(8):
            col = i // 4  
            row_frame = tk.Frame(folder_frame)
            row_frame.grid(row=i % 4, column=col, sticky="ew", padx=5, pady=1)
            
            btn = tk.Button(row_frame, text=f"Folder {i+1}", 
                          command=lambda idx=i: self.pick_folder(idx), width=10)
            btn.pack(side="left", padx=2)
            
            lbl = tk.Label(row_frame, text="[none]", font=("Arial", 7), anchor="w")
            lbl.pack(side="left", padx=5, fill="x", expand=True)
            self.folder_labels.append(lbl)
        
        folder_frame.columnconfigure(0, weight=1)
        folder_frame.columnconfigure(1, weight=1)

        btn_frame = tk.Frame(top)
        btn_frame.pack(fill="x", pady=3)
        
        self.btn_process = tk.Button(btn_frame, text="Process All Folders", 
                                     command=self.process_all, state=tk.DISABLED,
                                     font=("Arial", 8, "bold"), bg="#4CAF50", fg="white")
        self.btn_process.pack(side="left", padx=3)
        
        self.btn_save = tk.Button(btn_frame, text="Save All Results", 
                                 command=self.save_results, state=tk.DISABLED,
                                 font=("Arial", 8))
        self.btn_save.pack(side="left", padx=3)
        
        self.btn_random = tk.Button(btn_frame, text="Generate 5 Random Samples from Sphere", 
                                   command=self.generate_random_samples, state=tk.DISABLED,
                                   font=("Arial", 8), bg="#9C27B0", fg="white")
        self.btn_random.pack(side="left", padx=3)
        
        # NEW: Face Ranking Button
        self.btn_rank = tk.Button(btn_frame, text="Rank All Images by Reference Face", 
                                 command=self.rank_by_reference_face, state=tk.DISABLED,
                                 font=("Arial", 8, "bold"), bg="#FF5722", fg="white")
        self.btn_rank.pack(side="left", padx=3)

        self.lbl_status = tk.Label(top, text="Waiting for folders...", 
                                  font=("Arial", 7), fg="blue", 
                                  wraplength=1550, justify="left")
        self.lbl_status.pack(anchor="w", pady=2)

        self.progress = ttk.Progressbar(top, mode='determinate', length=300)
        self.progress.pack(anchor="w", pady=2)

        ttk.Separator(self.master, orient='horizontal').pack(fill='x', padx=8, pady=3)

        results_label = tk.Label(self.master, text="Results (Each Folder's Average + Global Average):", 
                                font=("Arial", 9, "bold"))
        results_label.pack(anchor="w", padx=8, pady=(2, 0))

        canvas_frame = tk.Frame(self.master)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=3)

        canvas = tk.Canvas(canvas_frame, bg="gray90", height=350)
        scrollbar_v = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar_h = tk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        self.img_frame = tk.Frame(scrollable_frame)
        self.img_frame.pack(padx=5, pady=5)

        self.img_labels = []
        for i in range(9):  # 8 folders + 1 global
            row = i // 3
            col = i % 3
            
            container = tk.Frame(self.img_frame, relief="raised", borderwidth=1)
            container.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            
            if i < 8:
                title = f"Folder {i+1} Avg"
                bg_color = "#2196F3"
            else:
                title = "GLOBAL Average"
                bg_color = "#FF5722"
            
            title_lbl = tk.Label(container, text=title, 
                               font=("Arial", 8, "bold"), 
                               bg=bg_color, 
                               fg="white", pady=2)
            title_lbl.pack(fill="x")
            
            img_lbl = tk.Label(container, bg="gray85", width=200, height=200)
            img_lbl.pack(padx=3, pady=3)
            
            info_lbl = tk.Label(container, text="", font=("Arial", 6), fg="gray30")
            info_lbl.pack()
            
            self.img_labels.append((img_lbl, info_lbl))

        ttk.Separator(self.master, orient='horizontal').pack(fill='x', padx=8, pady=3)

        random_label = tk.Label(self.master, text="Random Samples from PCA Sphere:", 
                               font=("Arial", 9, "bold"))
        random_label.pack(anchor="w", padx=8, pady=(2, 0))

        self.random_frame = tk.Frame(self.master, bg="gray90")
        self.random_frame.pack(fill="x", padx=8, pady=3)

        self.random_img_labels = []
        for i in range(5):
            container = tk.Frame(self.random_frame, relief="raised", borderwidth=1)
            container.pack(side="left", padx=3, pady=3)
            
            title_lbl = tk.Label(container, text=f"Sample {i+1}", 
                               font=("Arial", 7, "bold"), 
                               bg="#9C27B0", fg="white", pady=1)
            title_lbl.pack(fill="x")
            
            img_lbl = tk.Label(container, bg="gray85", width=140, height=140)
            img_lbl.pack(padx=2, pady=2)
            
            info_lbl = tk.Label(container, text="", font=("Arial", 6), fg="gray30")
            info_lbl.pack()
            
            self.random_img_labels.append((img_lbl, info_lbl))

    def pick_folder(self, idx):
        p = filedialog.askdirectory(title=f"Select Folder {idx+1}")
        if p:
            self.folders[idx] = p
            self.folder_labels[idx].config(text=os.path.basename(p))
            self._check_ready()

    def _check_ready(self): #Enable process button if all 8 folders are selected
        if all(f is not None for f in self.folders):
            self.btn_process.config(state=tk.NORMAL)
            self.lbl_status.config(text="All 8 folders selected. Click 'Process All Folders' to begin.")

    def process_all(self): #process 8 folders and creat global avg out of all items
        try:
            self.lbl_status.config(text="Processing folders...")
            self.progress['value'] = 0
            self.master.update_idletasks()
            all_items = []  
            self.all_items_info = []  # Reset
            
            # Process each folder
            for i, folder in enumerate(self.folders):
                self.lbl_status.config(text=f"Processing Folder {i+1}/8: {os.path.basename(folder)}")
                self.progress['value'] = (i / 9) * 100
                self.master.update_idletasks()
                
                # Load images
                items = load_folder_vectors(folder)
                if len(items) < 2:
                    messagebox.showwarning("Warning", 
                        f"Folder {i+1} has less than 2 images. Skipping.")
                    continue
                
                # Store items with folder information
                for fpath, vec in items:
                    self.all_items_info.append((fpath, vec, i))  # (filepath, vector, folder_index)
                
                all_items.extend(items)
                
                # Build data matrix (D, N)
                data_matrix = np.stack([vec for _, vec in items], axis=1)
                
                # Compute PCA
                k = min(K_CAP, data_matrix.shape[1] - 1, data_matrix.shape[0])
                mean_vec, eigvecs = compute_pca(data_matrix, k)
                
                # Project all images to PCA space
                coeffs_list = []
                for _, vec in items:
                    coeffs = project_to_pca(vec, mean_vec, eigvecs)
                    coeffs_list.append(coeffs)
                
                # Average the PCA coefficients
                avg_coeffs = np.mean(coeffs_list, axis=0)
                
                # Reconstruct from averaged coefficients
                recon_vec = reconstruct_from_pca(avg_coeffs, mean_vec, eigvecs)
                recon_img = vec_to_img_uint8(recon_vec, TARGET_SIZE)
                
                # Store results
                self.folder_data[i] = (items, mean_vec, eigvecs, avg_coeffs, recon_img)
                
                # Display
                self._display_image(i, recon_img, len(items))

            # Process global average
            self.lbl_status.config(text="Computing global average from all images...")
            self.progress['value'] = 90
            self.master.update_idletasks()
            
            if len(all_items) < 2:
                raise ValueError("Not enough total images across all folders.")
            
            # Build global data matrix
            global_matrix = np.stack([vec for _, vec in all_items], axis=1)
            
            # Compute global PCA
            k_global = min(K_CAP, global_matrix.shape[1] - 1, global_matrix.shape[0])
            global_mean, global_eigvecs = compute_pca(global_matrix, k_global)
            
            # Project all images to global PCA space
            global_coeffs_list = []
            for _, vec in all_items:
                coeffs = project_to_pca(vec, global_mean, global_eigvecs)
                global_coeffs_list.append(coeffs)
            
            # Average all coefficients
            global_avg_coeffs = np.mean(global_coeffs_list, axis=0)
            
            # Store all coefficients for sphere sampling
            self.all_global_coeffs = np.array(global_coeffs_list)  # Shape: (N, k)
            
            # Reconstruct global average
            global_recon_vec = reconstruct_from_pca(global_avg_coeffs, global_mean, global_eigvecs)
            global_recon_img = vec_to_img_uint8(global_recon_vec, TARGET_SIZE)
            
            self.global_result = (global_mean, global_eigvecs, global_avg_coeffs, global_recon_img)
            
            # Display global result
            self._display_image(8, global_recon_img, len(all_items))
            
            self.progress['value'] = 100
            self.lbl_status.config(text=f"✓ Complete! Processed {len(all_items)} total images across 8 folders.")
            self.btn_save.config(state=tk.NORMAL)
            self.btn_random.config(state=tk.NORMAL)
            self.btn_rank.config(state=tk.NORMAL)  # Enable ranking button
            
            messagebox.showinfo("Success", 
                f"Processing complete!\n\n"
                f"• 8 folder averages created\n"
                f"• 1 global average from {len(all_items)} images\n\n"
                f"PCA space has {self.all_global_coeffs.shape[1]} dimensions\n"
                f"Click 'Rank All Images' to compare with a reference face!")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            self.lbl_status.config(text=f"Error: {str(e)}")

    def _display_image(self, idx, img, num_images):
        pil_img = Image.fromarray(img)
        pil_img_display = pil_img.resize((190, 190), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img_display)
        
        self.photos[idx] = photo
        self.img_labels[idx][0].config(image=photo)
        self.img_labels[idx][1].config(text=f"{num_images} imgs")

    def save_results(self):
        try:
            out_dir = filedialog.askdirectory(title="Select Output Directory for Results")
            if not out_dir:
                return
            
            # Create subdirectory
            save_dir = os.path.join(out_dir, "pca_averages_output")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save each folder's average
            for i, data in enumerate(self.folder_data):
                if data is None:
                    continue
                _, _, _, _, recon_img = data
                filename = f"folder_{i+1}_average.png"
                cv2.imwrite(os.path.join(save_dir, filename), recon_img)
            
            # Save global average
            if self.global_result:
                _, _, _, global_img = self.global_result
                cv2.imwrite(os.path.join(save_dir, "global_average_all_images.png"), global_img)
            
            self.lbl_status.config(text=f"✓ All images saved to: {save_dir}")
            messagebox.showinfo("Saved", f"All results saved to:\n{save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")

    def generate_random_samples(self):
        try:
            if self.global_result is None or self.all_global_coeffs is None:
                messagebox.showerror("Error", "Process folders first!")
                return

            global_mean, global_eigvecs, _, _ = self.global_result
            
            # Calculate the center (mean), min, and max of the sphere in PCA space
            center = np.mean(self.all_global_coeffs, axis=0)  # Center of sphere
            min_vals = np.min(self.all_global_coeffs, axis=0)  # Minimum bounds
            max_vals = np.max(self.all_global_coeffs, axis=0)  # Maximum bounds
            
            #STD
            std_vals = np.std(self.all_global_coeffs, axis=0)
            
            self.lbl_status.config(text="Generating 5 random samples from PCA sphere...")
            self.master.update_idletasks()
            
            random_samples = []
            
            #5 random samples generation
            for i in range(5):
                # Method 1: Sample uniformly within min/max bounds (rectangular sampling)
                # random_coeffs = np.random.uniform(min_vals, max_vals)
                
                # Method 2: Sample from Gaussian distribution around center (more realistic)
                # This respects the actual distribution of faces in PCA space
                random_coeffs = np.random.normal(center, std_vals * 0.8)
                
                # Optional: Clip to stay within observed bounds
                random_coeffs = np.clip(random_coeffs, min_vals, max_vals)
                
                # Reconstruct image from random PCA coefficients
                recon_vec = reconstruct_from_pca(random_coeffs, global_mean, global_eigvecs)
                recon_img = vec_to_img_uint8(recon_vec, TARGET_SIZE)
                
                random_samples.append(recon_img)
                
                # Display
                pil_img = Image.fromarray(recon_img)
                pil_img_display = pil_img.resize((130, 130), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img_display)
                
                self.random_photos[i] = photo
                self.random_img_labels[i][0].config(image=photo)
                self.random_img_labels[i][1].config(text=f"{len(center)}D")
            
            # Save random samples
            if self.folders[0]:
                out_dir = os.path.join(os.path.dirname(self.folders[0]), "pca_random_samples")
                os.makedirs(out_dir, exist_ok=True)
                for i, img in enumerate(random_samples):
                    cv2.imwrite(os.path.join(out_dir, f"random_sample_{i+1}.png"), img)
                
                self.lbl_status.config(text=f"✓ Generated 5 random samples! Saved to: {out_dir}")
            else:
                self.lbl_status.config(text=f"✓ Generated 5 random samples from {len(center)}D PCA sphere!")
            
            messagebox.showinfo("Random Samples Generated", 
                f"Generated 5 random face samples!\n\n"
                f"PCA Space: {len(center)} dimensions\n"
                f"Center: Mean of all {self.all_global_coeffs.shape[0]} images\n"
                f"Sampling method: Gaussian around center\n\n"
                f"These are synthetic faces created by randomly\n"
                f"sampling the learned face space.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Random sampling failed:\n{str(e)}")

    def rank_by_reference_face(self):
        """NEW FEATURE: Rank all 50 images by distance to a reference face"""
        try:
            if self.global_result is None:
                messagebox.showerror("Error", "Process folders first!")
                return
            
            # Ask user to select reference face image
            ref_path = filedialog.askopenfilename(
                title="Select Reference Face Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
            )
            
            if not ref_path:
                return
            
            self.lbl_status.config(text="Processing reference face and ranking images...")
            self.progress['value'] = 0
            self.master.update_idletasks()
            
            # Load and preprocess reference image
            ref_bgr = cv2.imread(ref_path)
            if ref_bgr is None:
                messagebox.showerror("Error", "Cannot read reference image!")
                return
            
            ref_gray = preprocess_bgr(ref_bgr)
            ref_vec = ref_gray.flatten().astype(np.float32)
            
            # Project reference face to global PCA space
            global_mean, global_eigvecs, _, _ = self.global_result
            ref_coeffs = project_to_pca(ref_vec, global_mean, global_eigvecs)
            
            self.progress['value'] = 30
            self.master.update_idletasks()
            
            # Calculate distances to all images
            distances = []
            for fpath, vec, folder_idx in self.all_items_info:
                # Project image to PCA space
                img_coeffs = project_to_pca(vec, global_mean, global_eigvecs)
                
                # Calculate Euclidean distance in PCA space
                distance = np.linalg.norm(ref_coeffs - img_coeffs)
                
                # Extract person name and expression
                folder_name = os.path.basename(self.folders[folder_idx])
                img_name = os.path.splitext(os.path.basename(fpath))[0]
                
                # Format: PersonName_Expression
                label = f"{folder_name}_{img_name}"
                
                distances.append({
                    'distance': distance,
                    'label': label,
                    'filepath': fpath,
                    'folder': folder_name,
                    'image': img_name
                })
            
            self.progress['value'] = 70
            self.master.update_idletasks()
            
            # Sort by distance (closest first)
            distances.sort(key=lambda x: x['distance'])
            
            # Create DataFrame for Excel export
            df_data = []
            for rank, item in enumerate(distances, start=1):
                df_data.append({
                    'Rank': rank,
                    'Image': item['label']
                })
            
            df = pd.DataFrame(df_data)
            
            self.progress['value'] = 90
            self.master.update_idletasks()
            
            # Ask user where to save Excel file
            save_path = filedialog.asksaveasfilename(
                title="Save Ranking Results",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
            
            if save_path:
                # Save to Excel
                df.to_excel(save_path, index=False, sheet_name='Face Rankings')
                
                self.progress['value'] = 100
                self.lbl_status.config(text=f"✓ Ranking complete! Results saved to: {save_path}")
                
                # Show summary
                top_5 = "\n".join([f"{i+1}. {distances[i]['label']} (dist: {distances[i]['distance']:.2f})" 
                                  for i in range(min(5, len(distances)))])
                
                messagebox.showinfo("Ranking Complete", 
                    f"Successfully ranked {len(distances)} images!\n\n"
                    f"Reference: {os.path.basename(ref_path)}\n\n"
                    f"Top 5 closest matches:\n{top_5}\n\n"
                    f"Full results saved to:\n{save_path}")
            else:
                self.lbl_status.config(text="Ranking cancelled - file not saved.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Ranking failed:\n{str(e)}")
            self.lbl_status.config(text=f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EightFolderPCAApp(root)
    root.mainloop()