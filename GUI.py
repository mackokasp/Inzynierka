from tkinter import Tk, Label, Button,Entry,OptionMenu,StringVar
import tkinter as tk
import  Finance as fn
import Graph as gp
import Optimizer as opt
import copy
from PIL import ImageTk, Image

class MyFirstGUI:



    def __init__(self, mw):
        self.num =15
        self.mw = mw
        mw.title("Optymalizacja omega")
        mw.geometry("700x700")  # You want the size of the app to be 700x700
        mw.resizable(0, 0)
        self.mystring = tk.StringVar(mw,'100')
        self.mystring2 = tk.StringVar(mw,'0')
        self.weights=[]
        self.tickers=[]
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
        self.var.set('srednia')  # initial value
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
        self.tickers[0].set('AAN')
        self.tickers[1].set('CNP')
        self.tickers[2].set('F')
        self.tickers[3].set('GE')
        self.tickers[4].set('WMT')
        Button(mw, text="Optymalizuj", command=self.optimize).place(x=120, y=550, width=100)
        self.graph_btn=Button(mw, text="Pokaż wykres",state=tk.DISABLED, command=self.draw)
        self.table_btn = Button(mw, text="Rysuj tabele", state=tk.DISABLED, command=self.table)
        self.eval_button = Button(mw,text='Porównaj pozycje',command=self.eval_portfolio)
        self.graph_btn.place(x=220,y=550,width=100)
        self.table_btn.place(x=320, y=550, width=100)
        self.eval_button.place(x=420, y=550, width=150)

        self.yearfrom = tk.StringVar(mw)
        self.method = tk.StringVar(mw)
        self.yearfrom.set('2000')  # initial value
        self.option = OptionMenu(mw, self.yearfrom, "1980", "1990", "1995", "2000", "2005", "2010", "2015").place(x=470, y=75, width=80)
        self.option = OptionMenu(mw, self.method, "SLSQP", "lin").place(x=510,y=115,width=80)
        self.yearto = tk.StringVar(mw)
        self.yearto.set('2017')  # initial value
        self.method.set('SLSQP')
        self.option = OptionMenu(mw, self.yearto, "2000", "2005", "2010","2015","2016" ,"2017").place(x=570,y=75,width=80)
        Label(text="Dane od roku:" ).place(x=390, y=80)
        Label(text="Metoda Optymalizacji:").place(x=390, y=125)
        Label(text="do",).place(x=550, y=80)
        self.status =tk.StringVar(mw,'Wprowadz dane do optymalizacji !')
        Entry(textvariable=self.status ,state=tk.DISABLED).place(x=225, y=600, width=300)
        Label(text="Status:",font='Times').place(x=160, y=595)
        '''
        img = ImageTk.PhotoImage(Image.open('pobrane.jpg'))
        self.panel= Label(mw,image=img)
        self.panel.image=img

        #self.panel.pack(side="bottom", fill="both", expand="yes")
        self.panel.place(x=500, y=250,width=300,height=400)
         '''




    def greet(self):
        print(self.mystring.get())

    def draw(self):

        gp.draw_portfolios_omega(self.returns ,self.tick,self.sol)

    def table(self):
        gp.draw_table(self.returns,self.sol)




    def eval_portfolio (self):
        self.tick=[]
        for i in range(len(self.tickers)):
            if self.tickers[i].get() != '':
                self.tick.append(self.tickers[i].get().upper())
        gp.eval_results4(self.tick,int(self.yearfrom.get()),int(self.yearto.get()))




    def optimize(self):
        self.status.set('Optymalizuje')
        self.tick =[]
        error=0
        maxs=self.mystring.get()
        mins=self.mystring2.get()
        try:
            max = float(maxs) / 100
            min = float(mins) / 100
            if max <0.5 :
                self.status.set('Za mała wartość maksymalnego udziału')
                error=1
            if min >0.1:
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
                    target = float(self.var.get()) / 100

                except ValueError:
                    self.var.set('Błąd!!')
                    self.status.set('Podano niepoprawny cel')

                    return



                fn.set_target(target/100)

            for i in range(len(self.tickers)):
                if self.tickers[i].get() != '':
                    self.tick.append(self.tickers[i].get().upper())
            try:
                if self.freq =='roczny':
                    self.returns = fn.year_returns(self.tick, int(self.yearfrom.get()), int(self.yearto.get()))
                else:
                    self.returns = fn.month_returns(self.tick, int(self.yearfrom.get()), int(self.yearto.get()))

            except:
                self.status.set('Podano niepoprawne ID firm')
                error = 1



        if error==0:
            opt.set_returns(self.returns)
            if (self.var.get()=='srednia'):
                fn.set_average(self.returns)
            self.sol = opt.optimize(ratio='omega', method=self.method.get(), minW=min, maxW=max)
            sol2 = copy.deepcopy(self.sol)
            tick = sorted(self.tick)
            for j in range(len(tick)):
                sol2[j] = sol2[j] * 100
                self.tickers[j].set(tick[j])
                self.weights[j].set('{0:.4f}'.format(sol2[j]))
            self.graph_btn.configure(state="normal")
            self.table_btn.configure(state="normal")
            if sol2 is not None :
                self.status.set('Zoptymalizowano !')
            else :
                self.status.set('Coś poszło nie tak')





















def start():
    root = Tk()
    root.title('Optymalizacja Omega')
    root.configure(background='white')
    my_gui = MyFirstGUI(root)
    root.mainloop()