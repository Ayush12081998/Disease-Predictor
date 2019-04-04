from tkinter import Tk, Label, Button, Checkbutton, IntVar, messagebox,Entry

class MyFirstGUI2:

    def __init__(self, master):
        self.master = master
        master.title("NEW USER REGISTER HERE")
        self.label = Label(master, text="NEW USER FILL FORM FIRST")
        self.label.grid(column=0, row=1,)

        self.label2 = Label(master, text="user_id :")
        self.label2.grid(column=0, row=2, sticky='e')
        self.login = Entry(master, text="select:")
        self.login.grid(column=5, row=2)

        self.label3 = Label(master, text="password :")
        self.label3.grid(column=0, row=4, sticky='e')
        self.password = Entry(master)
        self.password.grid(column=5, row=4)

        self.label4 = Label(master, text="Name :")
        self.label4.grid(column=0, row=6, sticky='e')
        self.name = Entry(master)
        self.name.grid(column=5, row=6)

        self.label5 = Label(master, text="Address :")
        self.label5.grid(column=0, row=8, sticky='e')
        self.add = Entry(master)
        self.add.grid(column=5, row=8)



        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(column=0, row=10)
        self.submit_button = Button(master, text="Submit", command=master.quit)
        self.submit_button.grid(column=2, row=10)



class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.title("Disease Predictor")
        self.label = Label(master, text="ENTER LOGIN ID AND PASSWORD")
        self.label.grid(column=0, row=1,)

        self.label2 = Label(master, text="login_id :")
        self.label2.grid(column=0, row=2, sticky='e')
        self.login = Entry(master, text="select:")
        self.login.grid(column=5, row=2)
        self.label3 = Label(master, text="password :")
        self.label3.grid(column=0, row=4, sticky='e')
        self.password = Entry(master, show="*")
        self.password.grid(column=5, row=4)


        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(column=0, row=6)
        self.login_button = Button(master, text="Login", command=self.connect)
        self.login_button.grid(column=1, row=6)
        self.register_button = Button(master, text="new register", command=self.new_user)
        self.register_button.grid(column=2, row=6)


    def new_user(self):
        root1=Tk()
        my_gui = MyFirstGUI2(root1)
        root1.mainloop()

    def connect(self):

        root1=Tk()
        mygui=MyFirstGUI3(root1)
        root1.mainloop()


class MyFirstGUI3:
    symptoms = []
    Sym = []
    symp_val = []
    i = 2
    j = 0
    k = 0
    result = ""
    def __init__(self, master):
        self.master = master
        master.title("Disease Predictor")

        self.dataloading()

        self.label = Label(master, text="select the problems you are facing:")
        self.label.grid(column=0, row=1)

        for feel in self.symptoms:
            self.symp_val.append(IntVar())
            self.l = Checkbutton(self.master, text=feel, variable=self.symp_val[self.k])
            self.l.grid(column=self.j, row=self.i, sticky='w')
            self.l.deselect()

            self.k = self.k+1
            if(self.j < 5):
                    self.j =self.j+1
            else:
                self.j = 0
                self.i = self.i + 1

        self.Diagnosis = Button(master, text="Start Diagnosis", command=self.diagnosis)
        self.Diagnosis.grid(column=10, row=2)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(column=10, row=3)

        """self.scrollbar = Scrollbar(master)
        self.scrollbar.grid(column=5)"""



    def mlalgo(self):
        import pandas as pd
        import numpy as np

        # importing data
        dataset = pd.read_csv('Training.csv')
        dataset1 = pd.read_csv('Testing.csv')

        # matrix of features
        X_train = dataset.iloc[:, 0:-1].values
        Y_train = dataset.iloc[:, 132]
        X_test = dataset1.iloc[:, 0:-1].values
        Y_test = dataset1.iloc[:, 132]

        # categorizing the data
        from sklearn.preprocessing import LabelEncoder
        lblenc_train = LabelEncoder()
        lblenc_test = LabelEncoder()
        lblenc_train.fit_transform(Y_train)
        lblenc_test.fit_transform(Y_test)

        from sklearn.tree import DecisionTreeClassifier
        # create naive byes object
        nB = DecisionTreeClassifier(criterion='entropy')

        # train the model using the training sets
        nB.fit(X_train, Y_train)

        # making predictions on the testing set
        y_pred = nB.predict(X_test)

        # processing user data to make it fit in algorithm
        arr = np.asarray(self.Sym)
        mat = np.zeros((132, 132), dtype=int)
        mat[0] = arr
        # print(mat)
        new_df = pd.DataFrame(columns=self.symptoms, data=mat)
        print(new_df)
        new_df1 = pd.DataFrame()
        new_df1 = new_df[0:1]
        print(new_df1)
        # making prediction on user data
        Y_check = nB.predict(new_df1)
        print(Y_check)
        self.result = Y_check[0]



    def diagnosis(self):
        print("keep calm nothing serious!")
        for i in range(0, 132):
            if self.symp_val[i].get() == 1:
                self.Sym[i] = 1
        self.disp()
        self.mlalgo()
        variable = messagebox.askquestion('your diaganosis', self.result)
        if variable:
            self.master.quit()


    def disp(self):
        print(len(self.Sym))


    def dataloading(self):
        import pandas as pd
        dataset = pd.read_csv('Training.csv')
        self.symptoms = list(dataset)
        self.symptoms = self.symptoms[:-1]
        print(len(self.symptoms))
        # crating numeric array which is checked in model
        for i in range(0, 132):
            #print(i)
            self.Sym.append(0)



root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()