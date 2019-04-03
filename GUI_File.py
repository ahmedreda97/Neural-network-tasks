import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, StringVar, BooleanVar

import deepNN as bk

window = tk.Tk()


class InputVars:
    HiddenLayers_number = int()
    Neurons_number = int()
    Eta = int()
    M = int()
    B = BooleanVar()
    ActivationFunction = BooleanVar()


In = InputVars()


def WindowSetting():
    window.geometry("800x400")
    window.configure(background='white')
    window.title('Iris flower classification')
    window.iconbitmap('iris_icon_2mC_icon.ico')

    return


def LabelsSetting():
    HLayers = tk.Label(window, text='Number of HiddenLayers', fg='DarkOrchid3', bg='light Goldenrod1',
                       font=("Helvetica"))
    HLayers.place(x=80.0, y=50.0)

    Neurons = tk.Label(window, text='Number of Neurons per Layer', fg='DarkOrchid3', bg='light Goldenrod1',
                       font=("Helvetica"))
    Neurons.place(x=450.0, y=50.0)

    eta = tk.Label(window, text='Learning Rate', fg='DarkOrchid3', bg='light Goldenrod1', font=("Helvetica"))
    eta.place(x=80.0, y=100.0)

    m = tk.Label(window, text='Number of Epochs', fg='DarkOrchid3', bg='light Goldenrod1', font=("Helvetica"))
    m.place(x=450.0, y=100.0)

    Bias = tk.Label(window, text='Have a Bias?', fg='DarkOrchid3', bg='light Goldenrod1', font=("Helvetica"))
    Bias.place(x=80.0, y=150.0)

    ActvFunc = tk.Label(window, text='Activation Function', fg='DarkOrchid3', bg='light Goldenrod1', font=("Helvetica"))
    ActvFunc.place(x=450.0, y=150.0)
    return


def TextBoxesSetting():
    sv1 = StringVar()
    sv1.trace("w", lambda name, index, mode, sv1=sv1: callback1(sv1))
    etaTb = tk.Entry(window, textvariable=sv1)
    etaTb.place(x=280, y=55)

    sv2 = StringVar()
    sv2.trace("w", lambda name, index, mode, sv2=sv2: callback2(sv2))
    etaTb = tk.Entry(window, textvariable=sv2, width=15)
    etaTb.place(x=680, y=55)

    sv3 = StringVar()
    sv3.trace("w", lambda name, index, mode, sv3=sv3: callback3(sv3))
    epochsTb = tk.Entry(window, textvariable=sv3)
    epochsTb.place(x=280, y=105)

    sv4 = StringVar()
    sv4.trace("w", lambda name, index, mode, sv4=sv4: callback4(sv4))
    epochsTb = tk.Entry(window, textvariable=sv4, width=15)
    epochsTb.place(x=680, y=105)


def RadioButtoSetting():
    tk.Radiobutton(window, text="Yes", variable=In.B, value=True, bg='white', fg='DarkOrchid3').place(x=280, y=155)
    tk.Radiobutton(window, text="No", variable=In.B, value=False, bg='white', fg='DarkOrchid3').place(x=340, y=155)
    return


def ComboxesSetting():
    F1cmb = ttk.Combobox(window, values=('Hyperbolic Tangent Sigmoid', 'Sigmoid'), width=15)
    F1cmb.bind("<<ComboboxSelected>>", func=SelectedFromCombobox)
    F1cmb.place(x=680, y=155)
    return


def ButtonsSetting():
    RunBtn = tk.Button(window, text='Run', command=RunFunction, width=6, height=1,
                       bg='light Goldenrod1', fg='DarkOrchid3', activebackground='DarkGoldenrod1', font=("Helvetica"))
    RunBtn.place(x=190, y=300)

    ConfMatrix_Btn = tk.Button(window, text='Calculate Confusion Matrix', command=CalculateConfusionMatrix, width=23,
                               height=1,
                               bg='light Goldenrod1', fg='DarkOrchid3', activebackground='DarkGoldenrod1',
                               font=("Helvetica"))
    ConfMatrix_Btn.place(x=280, y=300)

    AccuracyBtn = tk.Button(window, text='Accuracy', command=GetAccuracy, width=10, height=1,
                            bg='light Goldenrod1', fg='DarkOrchid3', activebackground='DarkGoldenrod1',
                            font=("Helvetica"))
    AccuracyBtn.place(x=520, y=300)
    return


def RunFunction():
    global obj
    #  train a model: param (rate, hasBias, epoch, numLayers, numNodesInLayers, activationFunction)
    obj = bk.iris(In.Eta,In.B.get(),In.M, In.HiddenLayers_number,In.Neurons_number,In.ActivationFunction)
    # obj.plottingAll()
    obj.train()
    obj.test()
    # obj.plotting()
    return


def CalculateConfusionMatrix():
    # print("Test Run")

    messagebox.showinfo("Confusion Matrix",obj.matrix)
    return


def GetAccuracy():
    # print("Test Run")
    messagebox.showinfo("Accuracy",obj.accuracy)
    return


##################Setting the variables functions######################
def callback1(Str):
    In.HiddenLayers_number = Str.get()
    return


def callback2(Str):
    In.Neurons_number = Str.get()
    return


def callback3(Str):
    In.Eta = Str.get()
    return


def callback4(Str):
    In.M = Str.get()
    return


def SelectedFromCombobox(event):
    x = event.widget.get()
    In.ActivationFunction = ActivationFunctionAssighnment(x)
    return


def ActivationFunctionAssighnment(X):
    Choice = BooleanVar()
    if X == 'Sigmoid':
        Choice = 0
    elif X == 'Hyperbolic Tangent Sigmoid':
        Choice = 1
    return Choice


#########################################################################


WindowSetting()
LabelsSetting()
TextBoxesSetting()
RadioButtoSetting()
ComboxesSetting()
ButtonsSetting()
window.mainloop()  # your code goes here