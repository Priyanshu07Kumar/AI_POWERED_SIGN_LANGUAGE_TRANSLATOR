import customtkinter as ctk

ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.title("SignAI GUI Test")
app.geometry("600x400")

label = ctk.CTkLabel(
    app,
    text="AI POWERED SIGN LANGUAGE DETECTION",
    font=("Arial", 22, "bold")
)

label.pack(pady=100)

app.mainloop()