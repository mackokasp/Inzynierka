from tkinter import Tk, Label, Button,Entry,OptionMenu,StringVar
import tkinter as tk

class MyFirstGUI:



    def __init__(self, mw):
        self.mw = mw
        mw.title("Optymalizacja omega")
        mw.geometry("700x700")  # You want the size of the app to be 500x500
        mw.resizable(0, 0)
        self.mystring = tk.StringVar(mw,'100')
        self.mystring2 = tk.StringVar(mw,'0')
        self.label = Label(mw, text="Optymalizacja omega" ,font='Helvatica')
        self.label.pack()
        self.labelmax = Label(text="Max Udział").place(x=25, y=25)
        self.labelmax = Label(text="%").place(x=110, y=45)
        self.labelmax = Label(text="%").place(x=110, y=25)
        self.inputmax = Entry( text="1",textvariable = self.mystring).place(x=90, y=25, width=20)
        self.labelmin = Label(text="Min Udział ").place(x=25, y=45)
        self.inputmin = Entry( textvariable = self.mystring2).place(x=90, y=45, width=20)
        self.greet_button = Button( text="Greet", command=self.greet)
        self.greet_button.pack()
        var = tk.StringVar(mw)
        var.set('0.1')  # initial value
        self.option = OptionMenu(mw, var, "0.1", "0.5", "1", "2","3","10").place(x=550,y=25,width=80)
        Label(text="Cel:",font='Times').place(x=520, y=30)
        Label(text="%", font='Times').place(x=640, y=30)
        self.close_button = Button(mw, text="Close", command=mw.quit)
        self.close_button.pack()

    def greet(self):
        print(self.mystring.get())
def start():
    root = Tk()
    my_gui = MyFirstGUI(root)
    root.mainloop()