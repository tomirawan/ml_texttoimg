from tkinter import *
import customtkinter as ctk

from PIL import ImageTk

from diffusers import StableDiffusionPipeline
import torch
from torch import autocast

# Create the app
app = Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40, width=512, text_font=("arial 20"), \
                        text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

model = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model, revision="fp16", \
                                            torch_dtype=torch.float16, \
                                            use_auth_token=True)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)


trigger = ctk.CTkButton(height=40, width=120, text_font="arial 20", \
                        text_color="white", fg_color="blue", \
                        command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)



app.mainloop()