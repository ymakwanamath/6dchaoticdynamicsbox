"""Graphical User Interface for Chaotic Image Cipher."""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image, decrypt_image

class CipherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Image Cipher System")
        self.root.geometry("450x250")
        self.secret_key = None

        # Key Status
        self.lbl_key = tk.Label(root, text="Status: No Key Loaded", fg="red")
        self.lbl_key.pack(pady=10)

        # Buttons
        tk.Button(root, text="1. Generate ML-KEM Key", command=self.generate_key, width=30).pack(pady=5)
        tk.Button(root, text="2. Encrypt Image", command=self.encrypt_file, width=30).pack(pady=5)
        tk.Button(root, text="3. Decrypt Image", command=self.decrypt_file, width=30).pack(pady=5)
        
        self.lbl_status = tk.Label(root, text="Ready.")
        self.lbl_status.pack(pady=15)

    def generate_key(self):
        pk, _ = generate_keypair("ml-kem-768")
        self.secret_key, _ = encapsulate(pk, "ml-kem-768")
        self.lbl_key.config(text="Status: 256-bit Secret Key Loaded", fg="green")
        messagebox.showinfo("Key Exchange", "ML-KEM Shared Secret successfully established.")

    def run_task(self, task_func, *args):
        """Run encryption/decryption in a separate thread to prevent UI freezing."""
        def wrapper():
            try:
                task_func(*args)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.lbl_status.config(text="Error occurred.")
        threading.Thread(target=wrapper).start()

    def encrypt_file(self):
        if not self.secret_key:
            messagebox.showwarning("Warning", "Generate a key first.")
            return
            
        file_path = filedialog.askopenfilename(title="Select Image to Encrypt")
        if file_path:
            p = Path(file_path)
            out_path = p.parent / f"[Enc]{p.stem}.png"
            self.lbl_status.config(text=f"Encrypting {p.name}...")
            
            def do_encrypt():
                encrypt_image(str(p), self.secret_key, str(out_path))
                self.lbl_status.config(text=f"Saved: {out_path.name}")
                messagebox.showinfo("Success", f"Encrypted to {out_path.name}")
                
            self.run_task(do_encrypt)

    def decrypt_file(self):
        if not self.secret_key:
            messagebox.showwarning("Warning", "Generate a key first.")
            return
            
        file_path = filedialog.askopenfilename(title="Select Encrypted Image")
        if file_path:
            p = Path(file_path)
            stem = p.stem.replace("[Enc]", "")
            out_path = p.parent / f"[Dec]{stem}.png"
            self.lbl_status.config(text=f"Decrypting {p.name}...")
            
            def do_decrypt():
                decrypt_image(str(p), self.secret_key, str(out_path))
                self.lbl_status.config(text=f"Saved: {out_path.name}")
                messagebox.showinfo("Success", f"Decrypted to {out_path.name}")
                
            self.run_task(do_decrypt)

if __name__ == "__main__":
    root = tk.Tk()
    app = CipherGUI(root)
    root.mainloop()