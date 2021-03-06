import copy
import tkinter as tk
from tkinter import Tk, Label, Button, Entry, OptionMenu, ttk

import numpy as np

import Controller as con


#from PIL import ImageTk, Image

class MyFirstGUI:
    ticks = sorted(['S', 'WMT', 'VZ', 'RHT', 'DIS', 'JPM', 'BA', 'AMZN', 'PEP', 'ORCL', 'DAL'])






    def __init__(self, mw):
        self.style = ttk.Style()
        print(self.style.theme_names())
        self.style.theme_use('vista')
        self.num =15
        self.mw = mw
        mw.title("Optymalizacja omega")
        mw.geometry("700x700")  # You want the size of the app to be 700x700
        mw.resizable(0, 0)
        self.mystring = tk.StringVar(mw,'100')
        self.mystring2 = tk.StringVar(mw,'0')
        self.evalext_str = tk.StringVar(mw, '1')
        self.weights=[]
        self.tickers=[]
        self.max_number = tk.StringVar(mw, 3)
        self.freq =tk.StringVar(mw,'roczny')
        self.label = Label(mw, text="Optymalizacja omega" ,font='Helvatica')
        self.label.pack()
        self.labelmax = Label(text="Max Udział").place(x=25, y=25)
        self.labelmax = Label(text="%").place(x=110, y=45)
        self.labelmax = Label(text="%").place(x=110, y=25)
        self.inputmax = Entry( text="1",textvariable = self.mystring).place(x=90, y=25, width=20)
        self.labelmin = Label(text="Min Udział ").place(x=25, y=45)
        self.inputmin = Entry( textvariable = self.mystring2).place(x=90, y=45, width=20)
        self.var = tk.StringVar(mw)
        self.var.set('25')  # initial value
        self.option=Entry(text="1", textvariable=self.var).place(x=550, y=30, width=60)
        self.option = OptionMenu(mw, self.freq, "roczny", "miesieczny").place(x=440,y=28,width=100)
        Label(text="Cel:",font='Times').place(x=400, y=30)
        Label(text="%", font='Times').place(x=640, y=30)
        self.close_button = Button(mw, text="Zamknij", command=mw.quit).place(y=650,x=450)
        for i in (range(self.num)):
            self.weights.append(tk.StringVar(mw,'0'))
            self.tickers.append(tk.StringVar(mw, ''))
            Entry(textvariable=self.weights[i], state=tk.DISABLED).place(x=250, y=100 + i * 30, width=30)
            Entry(textvariable=self.tickers[i]).place(x=200, y=100+i*30, width=50)
            Label(text="%").place(x=280, y=100+i*30)
            Label(text="Akcja "+str(i+1)).place(x=150, y=100 + i * 30)
            j=0
        for i in self.ticks:
            self.tickers[j].set(i)
            j=j+1



        Button(mw, text="Optymalizuj", command=self.optimize).place(x=120, y=550, width=100)
        self.graph_btn=Button(mw, text="Pokaż wykres",state=tk.DISABLED, command=self.draw)
        self.table_btn = Button(mw, text="Rysuj tabele", state=tk.DISABLED, command=self.table)
        self.eval_button = Button(mw, text='Zmiany akcji', command=self.eval_portfolio)
        self.evalext_button = Button(mw, text='Zmiany portfolio', command=self.evalext_portfolio, state=tk.DISABLED)
        self.evalext_entry = Entry(textvariable=self.evalext_str).place(x=560, y=525, width=20)
        Label(text="Pokaz", font='Times').place(x=510, y=525)
        Label(text="lat po okresie", font='Times').place(x=585, y=525)

        self.graph_btn.place(x=220,y=550,width=100)
        self.table_btn.place(x=320, y=550, width=100)
        self.eval_button.place(x=420, y=550, width=100)
        self.evalext_button.place(x=520, y=550, width=100)

        self.yearfrom = tk.StringVar(mw)
        self.method = tk.StringVar(mw)
        self.yearfrom.set('2010')  # initial value
        self.option = OptionMenu(mw, self.yearfrom, "1980", "1990", "1995", "2000", "2003", "2005", "2008", "2010",
                                 "2012", "2013", "2014", "2015").place(x=470, y=75, width=80)
        self.option = OptionMenu(mw, self.method, "SLSQP", "lin", "elin", "lin_lmt", "lin_safe",
                                 command=self.change_opt)
        self.option.place(x=510, y=115, width=80)
        self.yearto = tk.StringVar(mw)
        self.yearto.set('2016')  # initial value
        self.method.set('lin')
        self.option = OptionMenu(mw, self.yearto, "2000", "2003", "2005", "2008", "2010", "2012", "2015", "2016",
                                 "2017").place(x=570, y=75, width=80)
        Label(text="Dane od roku:" ).place(x=390, y=80)
        Label(text="Metoda Optymalizacji:").place(x=390, y=125)
        Label(text="do",).place(x=550, y=80)
        self.status =tk.StringVar(mw,'Wprowadz dane do optymalizacji !')
        Entry(textvariable=self.status ,state=tk.DISABLED).place(x=225, y=600, width=300)
        self.max_entry = Entry(mw, textvariable=self.max_number, state=tk.DISABLED)
        self.max_entry.place(x=510, y=160, width=50)
        Label(text="Limit pozycji:", font='Times').place(x=410, y=160)
        Label(text="Status:",font='Times').place(x=160, y=595)
        '''
        img = ImageTk.PhotoImage(Image.open('pobrane.jpg'))
        self.panel= Label(mw,image=img)
        self.panel.image=img

        #self.panel.pack(side="bottom", fill="both", expand="yes")
        self.panel.place(x=500, y=250,width=300,height=400)
         '''

    def change_opt(self, event):
        if (self.method.get() == 'lin_lmt' or self.method.get() == 'lin_safe'):
            self.max_entry.configure(state="normal")


        else:
            self.max_entry.configure(state=tk.DISABLED)
            a = 0






    def greet(self):
        print(self.mystring.get())

    def draw(self):

        con.draw_portfolios_omega(self.returns, self.tick, self.sol)

    def table(self):
        con.draw_table(self.returns, self.sol)




    def eval_portfolio (self):
        self.tick=[]
        for i in range(len(self.tickers)):
            if self.tickers[i].get() != '':
                self.tick.append(self.tickers[i].get().upper())
        con.eval_results(self.tick, int(self.yearfrom.get()), int(self.yearto.get()))

    def evalext_portfolio(self):
        self.tick = []
        yearsaft = self.evalext_str.get()

        years_int = 0
        error = 0
        try:
            years_int = int(yearsaft)
            if int(self.yearto.get()) + years_int > 2017:
                self.status.set('Nieprawidłowy okres (dane tylko do konca 2017)')
                error = 1

        except:
            self.status.set('Podano nieprawidłową liczbe lat okresu po')
            error = 1

        if error == 0:
            for i in range(len(self.tickers)):
                if self.tickers[i].get() != '':
                    self.tick.append(self.tickers[i].get().upper())
            try:
                con.eval_portfolio(self.tick, int(self.yearfrom.get()), int(self.yearto.get()), years_int, self.sol)
            except:
                self.status.set('Podano nieprawidłową ID firm lub okres')






    def optimize(self):
        self.status.set('Optymalizuje')
        self.tick =[]
        error=0
        maxs=self.mystring.get()
        mins=self.mystring2.get()
        Ks = self.max_number.get()
        try:
            K = int(Ks)
            max = float(maxs) / 100
            min = float(mins) / 100
            if max < 0.3:
                self.status.set('Za mała wartość maksymalnego udziału')
                error=1
            if min > 0.3:
                self.status.set('Za duża wartość minimalnego udziału')
                error=1

        except:
            self.status.set('Podano niepoprawne wartosci ograniczen ')
            error=1



        if error==0:


            target = self.var.get()


            if target!='srednia':
                try:
                    float(target)
                    target = float(self.var.get())

                except ValueError:
                    self.var.set('Błąd!!')
                    self.status.set('Podano niepoprawny cel')

                    return

                con.set_target(target / 100)


            for i in range(len(self.tickers)):
                if self.tickers[i].get() != '':
                    self.tick.append(self.tickers[i].get().upper())
            try:
                if self.freq.get() == 'roczny':
                    self.returns = con.year_returns(self.tick, int(self.yearfrom.get()), int(self.yearto.get()))
                else:
                    self.returns = con.month_returns(self.tick, int(self.yearfrom.get()), int(self.yearto.get()))

            except:
                self.status.set('Podano niepoprawne ID firm')
                error = 1



        if error==0:
            con.set_returns(self.returns)
            if (self.var.get()=='srednia'):
                con.set_average(self.returns)
            try:
                self.sol = con.optimize(ratio='omega', method=self.method.get(), minW=min, maxW=max, K=K)
            except:
                self.status.set('Błąd optymlizacji')
                error = 1

            if self.sol is None or len(self.sol) < 1:
                self.status.set('Błąd optymlizacji')
                error = 1

            if error == 0:
                sol2 = copy.deepcopy(self.sol)
                tick = sorted(self.tick)
                for j in range(len(tick)):
                    sol2[j] = sol2[j] * 100
                    self.tickers[j].set(tick[j])
                    self.weights[j].set('{0:.4f}'.format(sol2[j]))
                self.graph_btn.configure(state="normal")
                self.table_btn.configure(state="normal")
                self.evalext_button.configure(state="normal")
                for k in range(len(tick), len(self.tickers)):
                    self.weights[k].set('{0:.4f}'.format(0.00))

                if sol2[1] is not None and not np.isnan(sol2[1]):
                    self.status.set('Zoptymalizowano !')
                else:
                    self.status.set('nie udalo sie znaleźc rozwiazania')























def start():
    root = Tk()

    root.title('Optymalizacja Omega')
    root.configure(background='white')
    my_gui = MyFirstGUI(root)
    root.mainloop()