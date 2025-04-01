import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def find_pictures():
    folder_path = '/Users/wojtek/repos/LabyCpsio/lab1/obrazy'
    return [f for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp','.tif'))]

def show_image():
    selected_file = dropdown.get()
    if not selected_file:
        return
    full_path = os.path.join(folder_path, selected_file)
    image = Image.open(full_path)
    image = image.resize((400, 400))  # skalowanie do rozmiaru okna
    photo = ImageTk.PhotoImage(image)
    label_image.config(image=photo)
    label_image.image = photo  # zachowaj referencję

if __name__ == "__main__":
    folder_path = '/Users/wojtek/repos/LabyCpsio/lab1/obrazy'
    pictures = find_pictures()

    root = tk.Tk()
    root.title("Wybór obrazu")

    ttk.Label(root, text="Wybierz obraz:").pack(pady=5)

    dropdown = ttk.Combobox(root, values=pictures, state="readonly")
    dropdown.pack(pady=5)

    btn_show = ttk.Button(root, text="Pokaż obraz", command=show_image)
    btn_show.pack(pady=5)

    label_image = ttk.Label(root)
    label_image.pack(pady=10)

    root.mainloop()
