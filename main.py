import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import shutil 
from keras.models import load_model
import pyperclip
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Couleurs: "blue" (standard), "green", "dark-blue"

class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("FPS OCR")
        self.geometry("700x500")
        self.resizable(False, True)
        
        # Création de Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both")
        
        # Tab "Choisir image"
        self.tab1 = self.tabview.add("Choisir image")
        self.add_choose_image_tab(self.tab1)
        
        # Tab "Binarisation"
        self.tab2 = self.tabview.add("Binarisation")
        self.add_binarization_tab(self.tab2)
        
        # Tab "Filtrage"
        self.tab3 = self.tabview.add("Filtrage")
        self.add_filtering_tab(self.tab3)
        
        # Tab "Segmentation"
        self.tab4 = self.tabview.add("Segmentation")
        self.add_segmentation_tab(self.tab4)
        
        # Tab "OCR"
        self.tab5 = self.tabview.add("OCR")
        self.add_ocr_tab(self.tab5)

        self.original_image = None  # Pour stocker l'image originale chargée
        self.filepath = None  # Chemin de l'image
        self.segmented_image = None  # Pour stocker l'image segmentée
        self.binarized_image = None  # Pour stocker l'image binarisée
    def add_choose_image_tab(self, tab):
        # Encapsuler le bouton et l'étiquette dans un cadre
        self.frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")  
        self.frame.pack(pady=20, padx=20, fill="both")
        
        self.choose_image_button = ctk.CTkButton(self.frame, text="Charger Image", command=self.load_image)
        self.choose_image_button.pack(pady=10)
        
        self.file_path_label = ctk.CTkLabel(self.frame, text="")
        self.file_path_label.pack(pady=5)
        
        self.image_label = ctk.CTkLabel(tab, text="Aucune image chargée")
        self.image_label.pack(pady=20)
    def add_binarization_tab(self, tab):
        self.binarization_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.binarization_frame.pack(pady=20, padx=20, fill="x")
        
        self.binarization_var = ctk.StringVar(value="global")
        
        self.global_binarization_radio = ctk.CTkRadioButton(self.binarization_frame, text="Seuil global", variable=self.binarization_var, value="global", command=self.update_binarization_options)
        self.global_binarization_radio.pack(pady=10, anchor='w')
        
        self.fixed_binarization_radio = ctk.CTkRadioButton(self.binarization_frame, text="Seuil fixe", variable=self.binarization_var, value="fixed", command=self.update_binarization_options)
        self.fixed_binarization_radio.pack(pady=10, anchor='w')
        
        self.fixed_threshold_entry = ctk.CTkEntry(self.binarization_frame, placeholder_text="Entrer le seuil fixe")
        self.fixed_threshold_entry.pack(pady=5)
        self.fixed_threshold_entry.configure(state='disabled')
        
        self.local_binarization_radio = ctk.CTkRadioButton(self.binarization_frame, text="Seuil local", variable=self.binarization_var, value="local", command=self.update_binarization_options)
        self.local_binarization_radio.pack(pady=10, anchor='w')
        
        self.local_blocksize_entry = ctk.CTkEntry(self.binarization_frame, placeholder_text="Taille de voisinage")
        self.local_blocksize_entry.pack(pady=5)
        self.local_blocksize_entry.configure(state='disabled')
        
        self.apply_binarization_button = ctk.CTkButton(self.binarization_frame, text="Appliquer binarisation", command=self.apply_binarization)
        self.apply_binarization_button.pack(pady=10)
        
        self.binarized_image_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.binarized_image_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.binarized_image_label = ctk.CTkLabel(self.binarized_image_frame, text="Aucune image binarisée")
        self.binarized_image_label.pack(pady=20)
        
        self.save_binarized_button = ctk.CTkButton(tab, text="Enregistrer Image Binarisée", command=self.save_binarized_image)
        self.save_binarized_button.pack(pady=10)

    def update_binarization_options(self):
        method = self.binarization_var.get()
        self.fixed_threshold_entry.configure(state='normal' if method == "fixed" else 'disabled')
        self.local_blocksize_entry.configure(state='normal' if method == "local" else 'disabled')

    def apply_binarization(self):
        if self.original_image is None:
            self.show_error("Aucune image chargée.")
            return

        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        method = self.binarization_var.get()
        binarized_image = None

        if method == "global":
            binarized_image = self.global_threshold(gray_image)
        elif method == "fixed":
            binarized_image = self.fixed_threshold(gray_image)
        elif method == "local":
            binarized_image = self.local_threshold(gray_image)

        if binarized_image is not None:
            self.display_binarized_image(binarized_image)

    def global_threshold(self, gray_image):
        _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized_image

    def fixed_threshold(self, gray_image):
        threshold = self.fixed_threshold_entry.get()
        if not threshold.isdigit():
            self.show_error("Veuillez entrer un seuil valide (un nombre entier).")
            return None
        threshold = int(threshold)
        if threshold < 0 or threshold > 255:
            self.show_error("Le seuil doit être un nombre entier entre 0 et 255.")
            return None
        _, binarized_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binarized_image

    def local_threshold(self, gray_image):
        blocksize = self.local_blocksize_entry.get()
        if not blocksize.isdigit() or int(blocksize) % 2 == 0:
            self.show_error("Veuillez entrer une taille de voisinage impaire valide (un nombre entier impair).")
            return None
        blocksize = int(blocksize)
        binarized_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, 2)
        return binarized_image

    def display_binarized_image(self, binarized_image):
        binarized_image_path = "binarized_image.png"
        cv2.imwrite(binarized_image_path, binarized_image)
        self.binarized_image = binarized_image_path
        self.display_image(binarized_image_path, self.binarized_image_label)

    def save_binarized_image(self):
        if self.binarized_image is None:
            self.show_error("Aucune image binarisée à enregistrer.")
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.png")])
        if save_path:
            shutil.copyfile(self.binarized_image, save_path)
            messagebox.showinfo("Sauvegarde réussie", f"L'image binarisée a été enregistrée avec succès sous {save_path}.")
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
        if file_path:
            if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                messagebox.showerror("Erreur de fichier", "Le fichier sélectionné n'est pas une image valide.")
                return
            relative_path = os.path.relpath(file_path)  
            self.file_path_label.configure(text=f"Chemin du fichier: {relative_path}")
            self.original_image = cv2.imread(file_path)  
            self.display_image(file_path, self.image_label)
            self.filepath = file_path
            self.segmented_image = None 
    def show_error(self, message):
        messagebox.showerror("Erreur", message)


    def display_image(self, file_path, label):
        img = Image.open(file_path)
        self.img_tk = ImageTk.PhotoImage(img)
        label.configure(image=self.img_tk, text="")
        label.image = self.img_tk


    def add_filtering_tab(self, tab):
        # Frame pour les boutons de filtre
        self.filter_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.filter_frame.pack(pady=20, padx=20, fill="x")
        
        self.filter_var = ctk.StringVar(value="median")
        
        self.median_filter_radio = ctk.CTkRadioButton(self.filter_frame, text="Filtrage médiane", variable=self.filter_var, value="median")
        self.median_filter_radio.pack(pady=10, anchor='w')
        
        self.gaussian_filter_radio = ctk.CTkRadioButton(self.filter_frame, text="Filtrage Gaussian", variable=self.filter_var, value="gaussian")
        self.gaussian_filter_radio.pack(pady=10, anchor='w')
        
        self.mean_filter_radio = ctk.CTkRadioButton(self.filter_frame, text="Filtrage moyen", variable=self.filter_var, value="mean")
        self.mean_filter_radio.pack(pady=10, anchor='w')
        
        self.apply_filter_button = ctk.CTkButton(self.filter_frame, text="Appliquer filtre", command=self.apply_filter)
        self.apply_filter_button.pack(pady=10)
        
        # Frame pour afficher l'image filtrée
        self.filtered_image_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.filtered_image_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.filtered_image_label = ctk.CTkLabel(self.filtered_image_frame, text="Aucune image filtrée")
        self.filtered_image_label.pack(pady=20) 

        # Bouton pour enregistrer l'image filtrée
        self.save_filtered_button = ctk.CTkButton(tab, text="Enregistrer Image Filtrée", command=self.save_filtered_image)
        self.save_filtered_button.pack(pady=10)

    def apply_filter(self):
        if self.original_image is None:
            messagebox.showerror("Erreur", "Aucune image chargée.")
            return
        
        filter_type = self.filter_var.get()
        filtered_image = None
        
        if filter_type == "median":
            filtered_image = cv2.medianBlur(self.original_image, 5)
        elif filter_type == "gaussian":
            filtered_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        elif filter_type == "mean":
            filtered_image = cv2.blur(self.original_image, (5, 5))
        
        if filtered_image is not None:
            filtered_image_path = "filtered_image.png"
            cv2.imwrite(filtered_image_path, filtered_image)
            self.filtered_image_path = filtered_image_path

            self.display_image(filtered_image_path, self.filtered_image_label)

    def save_filtered_image(self):
        if self.filtered_image_path is None:
            messagebox.showerror("Erreur", "Aucune image filtrée à enregistrer.")
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.png")])
        if save_path:
            shutil.copyfile(self.filtered_image_path, save_path)
            messagebox.showinfo("Sauvegarde réussie", f"L'image filtrée a été enregistrée avec succès sous {save_path}.")
    
    def add_ocr_tab(self, tab):
        self.ocr_display_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.ocr_display_frame.pack(pady=20, padx=20, fill="both")

       # Ajouter le bouton "Afficher Image Segmentée"
        self.display_segmented_image_button = ctk.CTkButton(self.ocr_display_frame, text="Afficher Image Segmentée", command=self.display_segmented_image)
        self.display_segmented_image_button.grid(row=0, column=0, padx=(0, 10), pady=10)

        # Ajouter le bouton "Reconnaître texte"
        self.recognize_text_button = ctk.CTkButton(self.ocr_display_frame, text="Reconnaître texte", command=self.recognize_text)
        self.recognize_text_button.grid(row=0, column=1, padx=(10, 0), pady=10)

        # Configure the ocr_display_frame to center its content horizontally
        self.ocr_display_frame.columnconfigure(0, weight=1)  # Expand the first column
        self.ocr_display_frame.columnconfigure(1, weight=1)  # Expand the second column


        # Ajouter le label pour afficher le texte reconnu
        self.recognized_text_label = ctk.CTkLabel(self.ocr_display_frame, text="", height=10, width=50)
        self.recognized_text_label.grid(row=1, column=0, columnspan=2, pady=10, padx=10)

        # Ajouter le label pour afficher l'image segmentée
        self.segmented_image_label = ctk.CTkLabel(self.ocr_display_frame, text="Aucune image segmentée")
        self.segmented_image_label.grid(row=2, column=0, columnspan=2, pady=10, padx=10)
        # Bouton pour copier le texte reconnu dans le bloc note
        self.copy_to_clipboard_button = ctk.CTkButton(tab, text="Copier Texte", command=self.copy_text_to_clipboard)
        self.copy_to_clipboard_button.pack(pady=10)
        

    def copy_text_to_clipboard(self):
        if self.recognized_text_label.cget("text") == "":
            messagebox.showerror("Erreur", "Aucun texte reconnu.")
            return
        pyperclip.copy(self.recognized_text_label.cget("text"))
        messagebox.showinfo("Copié dans le presse-papier", "Le texte reconnu a été copié dans le presse-papier.")
        


    def display_segmented_image(self):
        if self.segmented_image is None:
            messagebox.showerror("Erreur", "Aucune image segmentée.")
            return
        
        self.display_image(self.segmented_image, self.segmented_image_label)
        
    def recognize_text(self):
        if self.filepath is None:
            messagebox.showerror("Erreur", "Aucune image chargée.")
            return

        character_images = self.segment_characters(self.filepath)

        recognized_text = ""
        for char_img in character_images:
            recognized_char = self.recognize_character(char_img)
            recognized_text += recognized_char

        self.recognized_text_label.configure(text=recognized_text)

    def recognize_character(self, character_image):
        model = load_model('model.h5')
        dict_word = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
        character_image = cv2.resize(character_image, (28, 28))
        character_image = character_image.astype('float32') / 255.0
        character_image = np.expand_dims(character_image, axis=0)
        character_image = np.expand_dims(character_image, axis=-1)
        prediction = model.predict(character_image)
        predicted_character = dict_word[np.argmax(prediction)]
        return predicted_character

    def segment_characters(self, image_path, margin=5):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        cntrs = sorted(cntrs, key=lambda c: cv2.boundingRect(c)[0])
        character_images = []
        for c in cntrs:
            x, y, w, h = cv2.boundingRect(c)
            x -= margin  
            y -= margin
            w += 2 * margin
            h += 2 * margin
            character_images.append(thresh[max(0, y):min(y+h, thresh.shape[0]), max(0, x):min(x+w, thresh.shape[1])])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return character_images
    
    def add_segmentation_tab(self, tab):
        self.segmentation_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.segmentation_frame.pack(pady=20, padx=20, fill="x")

        button_frame = ctk.CTkFrame(self.segmentation_frame)
        button_frame.pack(pady=10)

        self.apply_segmentation_button = ctk.CTkButton(button_frame, text="Appliquer Segmentation", command=self.apply_segmentation)
        self.apply_segmentation_button.pack(side='left', pady=10, padx=(0, 20))

        self.previous_image_button = ctk.CTkButton(button_frame, text="Image Précédente", command=self.show_previous_image)
        self.previous_image_button.pack(side='left', pady=10, padx=(0, 20))

        self.save_segmented_button = ctk.CTkButton(button_frame, text="Enregistrer Image Segmentée", command=self.save_segmented_image)
        self.save_segmented_button.pack(side='left', pady=10, padx=(0, 20))

        self.segmented_image_frame = ctk.CTkFrame(tab, corner_radius=10, fg_color="#333333")
        self.segmented_image_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.segmented_image_label1 = ctk.CTkLabel(self.segmented_image_frame, text="Aucune image segmentée")
        self.segmented_image_label1.pack(pady=20)

    

    def save_segmented_image(self):
        if self.segmented_image is None:
            messagebox.showerror("Erreur", "Aucune image segmentée à enregistrer.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.png")])
        if save_path:
            cv2.imwrite(save_path, self.segmented_image)
            messagebox.showinfo("Sauvegarde réussie", f"L'image segmentée a été enregistrée avec succès sous {save_path}.")


    def apply_segmentation(self):
        if self.original_image is None:
            messagebox.showerror("Erreur", "Aucune image chargée.")
            return

        if self.segmented_image is None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            cntrs = sorted(cntrs, key=lambda c: cv2.boundingRect(c)[0])
            
            for c in cntrs:
                x, y, w, h = cv2.boundingRect(c)
                x -= 5  
                y -= 5
                w += 2 * 5
                h += 2 * 5
                cv2.rectangle(self.original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            segmented_image_path = "segmented_image.png"
            cv2.imwrite(segmented_image_path, self.original_image)
            self.segmented_image = segmented_image_path  

        self.display_image(self.segmented_image, self.segmented_image_label1)

    
    def show_previous_image(self):
        if self.filepath is None:
            messagebox.showerror("Erreur", "Aucune image chargée.")
            return

        self.segmented_image_label.configure(image=None, text="")
        self.display_image(self.filepath, self.segmented_image_label1)

if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()
