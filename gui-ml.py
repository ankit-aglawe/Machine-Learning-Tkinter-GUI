from tkinter import *
from tkinter.filedialog import askopenfilename
import csv
import os
from tkinter import ttk
import time

gui = Tk()

gui.title('Machine Learning GUI')

gui.geometry('600x600')

progress_bar = ttk.Progressbar(orient = 'horizontal', length=600, mode='determinate')
progress_bar.grid(row=150, columnspan=3, pady =10)


def data():
    global filename
    filename = askopenfilename(initialdir='C:\\',title = "Select file")
    e1.delete(0, END)
    e1.insert(0, filename)
    #e1.config(text=filename)
    #print(filename)



    import pandas as pd
    global file1

    file1 = pd.read_csv(filename)

    global col
    col = list(file1.head(0))
    #print(col)

    for i in range(len(col)):
        box1.insert(i+1, col[i])

def X_values():

    values = [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(values))):
        box2.insert(i+1, values[i])
        box1.selection_clear(i+1, END)
    X_values.x1=[]
    for j in range(len(values)):X_values.x1.append(j)

    global x_size
    x_size = len(X_values.x1)
    print(x_size)


    print(X_values.x1)



def y_values():
    values= [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(values))):
        box3.insert(i+1, values[i])
    y_values.y1=[]
    for j in range(len(values)):y_values.y1.append(j)


    print(y_values.y1)


def clear():
    pass

def sol():
    progress()
   
    from sklearn.model_selection import cross_val_score,train_test_split,KFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from matplotlib import pyplot as plt
   



    X = file1.iloc[:,X_values.x1].values
    y = file1.iloc[:,y_values.y1].values

    y = y.reshape((-1,))

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    le =LabelEncoder()
    hotlist =[]
    
    for i in range(X.shape[1]):
        if isinstance(X[1,i], str):
            X[:,i] = le.fit_transform(X[:,i])
            hotlist.append(i)
            #print('hello')
    #print(X)
    #print(hotlist)

    onehot = OneHotEncoder(categorical_features=hotlist)
    X = onehot.fit_transform(X).toarray()

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    X = sc.fit_transform(X) 

  
    kfold=KFold(10,random_state=7)
    models=[]
    models.append(("KNN",KNeighborsClassifier()))
    models.append(("NB",GaussianNB()))
    #models.append(("LG",LogisticRegression()))
    models.append(("Tree",DecisionTreeClassifier()))
    #models.append(("SVM",SVC()))
    results=[]
    names=[]
    scoring='accuracy'
    for name,model in models:
	    kfold=KFold(n_splits=5,random_state=5) 
	    v=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
	    results.append(v)
	    names.append(name)
	    print(name)
	    print(v)
    fig=plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax=fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    #stop_progressbar()

def progress():
    progress_bar['maximum']=100

    for i in range(101):
        time.sleep(0.01)
        progress_bar['value'] = i
        progress_bar.update()

    progress_bar['value'] = 0
'''
def start_progressbar():
    a = progress()
    a.progress_bar.start()

def stop_progressbar():
    a = progress()
    a.progress_bar.stop()

def twofunc():
    progress()
    sol()
    
'''

l1=Label(gui, text='Select Data File')
l1.grid(row=0, column=0)
e1 = Entry(gui,text='')
e1.grid(row=0, column=1)

Button(gui,text='open', command=data).grid(row=0, column=2)

box1 = Listbox(gui,selectmode='multiple')
box1.grid(row=10, column=0)
Button(gui, text='Clear All',command=clear).grid(row=12,column=0)

box2 = Listbox(gui)
box2.grid(row=10, column=1)
Button(gui, text='Select X', command=X_values).grid(row=12,column=1)

box3 = Listbox(gui)
box3.grid(row=10, column=2) 
Button(gui, text='Select y', command=y_values).grid(row=12,column=2)

Button(gui, text='Solution', command=sol).grid(row=20, column=1)





gui.mainloop()
