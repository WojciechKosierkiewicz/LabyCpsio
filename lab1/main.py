import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

def find_pictures():
    return [f for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

def load_image():
    global current_image_np, original_image_np
    selected_file = dropdown.get()
    if not selected_file:
        return
    full_path = os.path.join(folder_path, selected_file)
    image = Image.open(full_path).convert('L')
    original_image_np = np.array(image)
    current_image_np = original_image_np.copy()

    w, h = image.size
    label_size.config(text=f"Rozmiar: {w} × {h} px")

    scale_row.config(to=h-1)
    scale_col.config(to=w-1)

    show_image(current_image_np)

def show_image(img_array):
    image = Image.fromarray(img_array)
    image_resized = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image_resized)
    label_image.config(image=photo)
    label_image.image = photo

def update_profile_row(value):
    if current_image_np is None:
        return
    index = int(value)
    line = current_image_np[index, :]
    plt.clf()
    plt.plot(line)
    plt.title(f'Profil poziomy (wiersz {index})')
    plt.xlabel("Pozycja")
    plt.ylabel("Poziom szarości")
    plt.grid(True)
    plt.pause(0.01)

def update_profile_col(value):
    if current_image_np is None:
        return
    index = int(value)
    line = current_image_np[:, index]
    plt.clf()
    plt.plot(line)
    plt.title(f'Profil pionowy (kolumna {index})')
    plt.xlabel("Pozycja")
    plt.ylabel("Poziom szarości")
    plt.grid(True)
    plt.pause(0.01)

def save_crop():
    if current_image_np is None:
        return
    try:
        raw = entry_crop.get().strip()
        x, y, w, h, filename = raw.split(',')
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped = current_image_np[y:y+h, x:x+w]
        Image.fromarray(cropped).save(os.path.join(folder_path, filename))
        print(f"Zapisano podobszar jako: {filename}")
    except Exception as e:
        print("Błąd przy zapisie podobszaru:", e)

def apply_transformation():
    global current_image_np
    if original_image_np is None:
        return

    option = transform_option.get()
    r = original_image_np.astype(np.float32)

    try:
        c = float(entry_c.get())
    except:
        c = 1.0
    try:
        m = float(entry_m.get())
    except:
        m = 128.0
    try:
        e = float(entry_e.get())
    except:
        e = 4.0

    if option == "Brak":
        current_image_np = original_image_np.copy()
    elif option == "Mnożenie przez stałą":
        current_image_np = np.clip(c * r, 0, 255).astype(np.uint8)
    elif option == "Transformacja logarytmiczna":
        current_image_np = np.clip(c * np.log1p(r), 0, 255).astype(np.uint8)
    elif option == "Zmiana kontrastu":
        current_image_np = (255 / (1 + (m / (r + 1e-5)) ** e)).astype(np.uint8)

    show_image(current_image_np)

# === GUI ===
folder_path = '/Users/wojtek/repos/LabyCpsio/lab1/obrazy'
current_image_np = None
original_image_np = None

root = tk.Tk()
root.title("Transformacje punktowe i analiza obrazu")

# Obraz
ttk.Label(root, text="Wybierz obraz:").pack(pady=5)
dropdown = ttk.Combobox(root, values=find_pictures(), state="readonly")
dropdown.pack(pady=5)

btn_load = ttk.Button(root, text="Pokaż obraz", command=load_image)
btn_load.pack(pady=5)

label_image = ttk.Label(root)
label_image.pack(pady=10)

label_size = ttk.Label(root, text="Rozmiar: -")
label_size.pack(pady=5)

# Suwaki profili
ttk.Label(root, text="Profil poziomy (wiersz):").pack()
scale_row = tk.Scale(root, from_=0, to=100, orient='horizontal', command=update_profile_row)
scale_row.pack(fill='x', padx=10)

ttk.Label(root, text="Profil pionowy (kolumna):").pack()
scale_col = tk.Scale(root, from_=0, to=100, orient='horizontal', command=update_profile_col)
scale_col.pack(fill='x', padx=10)

# Transformacje
ttk.Label(root, text="Transformacja punktowa:").pack(pady=10)
transform_option = ttk.Combobox(root, state="readonly", values=[
    "Brak",
    "Mnożenie przez stałą",
    "Transformacja logarytmiczna",
    "Zmiana kontrastu"
])
transform_option.set("Brak")
transform_option.pack(pady=5)

# Pola do ustawienia parametrów
frame_params = ttk.Frame(root)
frame_params.pack(pady=5)

ttk.Label(frame_params, text="c:").grid(row=0, column=0)
entry_c = ttk.Entry(frame_params, width=6)
entry_c.insert(0, "1.5")
entry_c.grid(row=0, column=1)

ttk.Label(frame_params, text="m:").grid(row=0, column=2)
entry_m = ttk.Entry(frame_params, width=6)
entry_m.insert(0, "128")
entry_m.grid(row=0, column=3)

ttk.Label(frame_params, text="e:").grid(row=0, column=4)
entry_e = ttk.Entry(frame_params, width=6)
entry_e.insert(0, "4")
entry_e.grid(row=0, column=5)

ttk.Button(root, text="Zastosuj transformację", command=apply_transformation).pack(pady=5)

# Podobraz
ttk.Label(root, text="x,y,szer,wys,nazwa_pliku").pack(pady=10)
entry_crop = ttk.Entry(root)
entry_crop.insert(0, "50,50,100,100,podobraz.png")
entry_crop.pack(pady=5)

ttk.Button(root, text="Zapisz podobszar", command=save_crop).pack(pady=5)

# matplotlib interaktywny
plt.ion()
plt.figure()

root.mainloop()
