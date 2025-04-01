import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

def find_pictures():
    return [f for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp','.tif'))]

def load_image():
    global current_image_np
    selected_file = dropdown.get()
    if not selected_file:
        return
    full_path = os.path.join(folder_path, selected_file)
    image = Image.open(full_path).convert('L')
    current_image_np = np.array(image)

    w, h = image.size
    label_size.config(text=f"Rozmiar: {w} × {h} px")

    image_resized = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image_resized)
    label_image.config(image=photo)
    label_image.image = photo

def show_profile(orientation):
    if current_image_np is None:
        return
    try:
        index = int(entry_coord.get())
        if orientation == 'horizontal':
            line = current_image_np[index, :]
            title = f'Profil poziomy (wiersz {index})'
        else:
            line = current_image_np[:, index]
            title = f'Profil pionowy (kolumna {index})'
        plt.figure()
        plt.plot(line)
        plt.title(title)
        plt.xlabel("Pozycja")
        plt.ylabel("Poziom szarości")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Błąd:", e)

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

# Ścieżka do katalogu z obrazami
folder_path = '/Users/wojtek/repos/LabyCpsio/lab1/obrazy'
current_image_np = None

root = tk.Tk()
root.title("Wybór obrazu, profil szarości i zapis podobszaru")

# Wybór obrazu
ttk.Label(root, text="Wybierz obraz:").pack(pady=5)
dropdown = ttk.Combobox(root, values=find_pictures(), state="readonly")
dropdown.pack(pady=5)


btn_load = ttk.Button(root, text="Pokaż obraz", command=load_image)
btn_load.pack(pady=5)

label_size = ttk.Label(root, text="Rozmiar: -")
label_size.pack(pady=5)  # 🔧 dodane

label_image = ttk.Label(root)
label_image.pack(pady=10)

# Profil szarości
ttk.Label(root, text="Wpisz współrzędną linii (wiersz/kolumna):").pack(pady=5)
entry_coord = ttk.Entry(root)
entry_coord.insert(0, "100")
entry_coord.pack(pady=5)

ttk.Button(root, text="Profil poziomy", command=lambda: show_profile('horizontal')).pack(pady=2)
ttk.Button(root, text="Profil pionowy", command=lambda: show_profile('vertical')).pack(pady=2)

# Podobraz
ttk.Label(root, text="x,y,szer,wys,nazwa_pliku").pack(pady=10)
entry_crop = ttk.Entry(root)
entry_crop.insert(0, "50,50,100,100,podobraz.png")
entry_crop.pack(pady=5)

ttk.Button(root, text="Zapisz podobszar", command=save_crop).pack(pady=5)

root.mainloop()
