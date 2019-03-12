import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox,StringVar,BooleanVar
import Network as bk

window = tk.Tk()
class InputVars:
    Ft1=int()
    Ft2=int()
    Class1 = int()
    Class2 = int()
    Eta = int()
    M = int()
    B=BooleanVar()
    mse=float()
In = InputVars()
#obj = bk.iris(In.Ft1,In.Ft2,In.Class1,In.Class2,In.Eta,In.B,In.M)

def WindowSetting():
    window.geometry("800x400")
    window.configure(background='white')
    window.title('Iris flower classification')
    window.iconbitmap('iris_icon_2mC_icon.ico')

    return

def LabelsSetting():
    F1= tk.Label(window, text='First Feature', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    F1.place(x=80.0 , y=50.0)
    F2= tk.Label(window, text='Second Feature', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    F2.place(x=450.0 , y=50.0)

    C1= tk.Label(window, text='First Class', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    C1.place(x=80.0 , y=100.0)
    C2= tk.Label(window, text='Second Class', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    C2.place(x=450.0 , y=100.0)

    eta= tk.Label(window, text='Learning Rate', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    eta.place(x=80.0 , y=150.0)

    m= tk.Label(window, text='Number of Epochs', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    m.place(x=450.0 , y=150.0)

    Bias= tk.Label(window, text='Have a Bias?', fg='DarkOrchid3',bg='light Goldenrod1',font=("Helvetica"))
    Bias.place(x=80.0 , y=200.0)

    Bias = tk.Label(window, text='mse_thres(ada)', fg='DarkOrchid3', bg='light Goldenrod1', font=("Helvetica"))
    Bias.place(x=450.0, y=200.0)

    return

def ButtonsSetting():
    RunBtn= tk.Button(window,text='Run',command=RunFunction, width=6 , height=1,
                     bg='light Goldenrod1',fg='DarkOrchid3',activebackground='DarkGoldenrod1',font=("Helvetica"))
    RunBtn.place(x=190,y=300)

    RunAdaBtn = tk.Button(window, text='RunAda', command=RunAdaFunction, width=6, height=1,
                       bg='light Goldenrod1', fg='DarkOrchid3', activebackground='DarkGoldenrod1', font=("Helvetica"))
    RunAdaBtn.place(x=90, y=300)

    ConfMatrix_Btn= tk.Button(window,text='Calculate Confusion Matrix',command=CalculateConfusionMatrix, width=23 , height=1,
                     bg='light Goldenrod1',fg='DarkOrchid3',activebackground='DarkGoldenrod1',font=("Helvetica"))
    ConfMatrix_Btn.place(x=280,y=300)

    AccuracyBtn= tk.Button(window,text='Accuracy',command=GetAccuracy, width=10 , height=1,
                     bg='light Goldenrod1',fg='DarkOrchid3',activebackground='DarkGoldenrod1',font=("Helvetica"))
    AccuracyBtn.place(x=520,y=300)
    return
def ComboxesSetting():
    F1cmb = ttk.Combobox(window,values=('X1','X2','X3','X4'))
    F1cmb.bind("<<ComboboxSelected>>",func=SelectedFromCombobox1)
    F1cmb.place(x=200,y=52)

    F2cmb = ttk.Combobox(window,values=('X1','X2','X3','X4'))
    F2cmb.bind("<<ComboboxSelected>>",func=SelectedFromCombobox2)
    F2cmb.place(x=595,y=52)

    C1cmb = ttk.Combobox(window,values=('Iris Setosa','Iris Versicolor','Iris Virginica'))
    C1cmb.bind("<<ComboboxSelected>>",func=SelectedFromCombobox3)
    C1cmb.place(x=200,y=102)

    C2cmb = ttk.Combobox(window, values=('Iris Setosa', 'Iris Versicolor', 'Iris Virginica'))
    C2cmb.bind("<<ComboboxSelected>>",func=SelectedFromCombobox4)
    C2cmb.place(x=595, y=102)

    return
#Events' Handlers to take the selected item from the combobox
def SelectedFromCombobox1(event):
    x=event.widget.get()
    In.Ft1 = FaetureAssighnment(x)
    return

def SelectedFromCombobox2(event):
    x=event.widget.get()
    In.Ft2 = FaetureAssighnment(x)
    return
def SelectedFromCombobox3(event):
    x=event.widget.get()
    In.Class1 = ClassAssignment(x)
    return
def SelectedFromCombobox4(event):
    x=event.widget.get()
    In.Class2 = ClassAssignment(x)
    return
def callback1(Str):
    In.Eta= Str.get()
    return
def callback2(Str):
    In.M= Str.get()
    return
def callback3(Str):
    In.mse=Str.get()
    return
################################################################
def TextBoxesSetting():
    sv= StringVar()
    sv.trace("w",lambda name, index, mode, sv=sv: callback1(sv))
    etaTb=tk.Entry(window,textvariable=sv)
    etaTb.place(x=200,y=153)

    sv2= StringVar()
    sv2.trace("w",lambda name, index, mode, sv2=sv2: callback2(sv2))
    epochsTb=tk.Entry(window,textvariable=sv2)
    epochsTb.place(x=595,y=153)

    sv3 = StringVar()
    sv3.trace("w", lambda name, index, mode, sv3=sv3: callback3(sv3))
    epochsTb = tk.Entry(window, textvariable=sv3)
    epochsTb.place(x=595, y=200)

    return
def RadioButtoSetting():

    tk.Radiobutton(window,text="Yes",variable=In.B,value=True,bg='white',fg='DarkOrchid3').place(x=200,y=200)
    tk.Radiobutton(window, text="No", variable=In.B, value=False,bg='white',fg='DarkOrchid3').place(x=250,y=200)
    return

def FaetureAssighnment(F):
    OF = int()
    if F == 'X1':
        OF=0
    elif F=='X2':
        OF=1
    elif F=='X3':
        OF=2
    else:
        OF=3
    return OF

def ClassAssignment(C):
    if C == 'Iris Setosa':
        OC=0
    elif C == 'Iris Versicolor':
        OC=1
    else:
        OC=2
    return OC
def RunFunction():
    global obj
    obj = bk.iris(In.Ft1,In.Ft2,In.Class1,In.Class2,In.Eta,In.B.get(),In.M)
    #obj.plottingAll()
    obj.train()
    obj.test()
    obj.plotting()
    return

def RunAdaFunction():
    global obj
    obj = bk.iris(In.Ft1,In.Ft2,In.Class1,In.Class2,In.Eta,In.B.get(),In.M,In.mse)
    #obj.plottingAll()
    obj.trainAda()
    obj.test()
    obj.plotting()
    return
def CalculateConfusionMatrix():
    messagebox.showinfo("Confusion Matrix",obj.matrix)
    return
def GetAccuracy():
     messagebox.showinfo("Accuracy",obj.accuracy)
     return

WindowSetting()
LabelsSetting()

ComboxesSetting()
TextBoxesSetting()
RadioButtoSetting()
ButtonsSetting()
window.mainloop()