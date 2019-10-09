#!/usr/bin/env python3
from tkinter import *
from tkinter import messagebox, filedialog, Menu
import tkinter.ttk as ttk
#from tkinter.ttk import *
import os
import constants as co
#os.chdir(os.path.dirname(__file__))

class q2mm_gui():
    def __init__(self):
        self.anchor = W
        
        self.cur_dir = os.getcwd()
        self.path = ["","","","",""] # ff file, qparam.txt, init_comparison, final ff file, fin_comparison
        self.weights = []
        #co.WEIGHTS
        self.qm_soft = [None,None,None,None,None,None,None]
        self.mm_soft = [None,None,None,None,None,None,None]
        self.open_f = None
        self.save_f = None
        self.commands = "DIR aa \nFFLD read bb \nWGHT cc \nPARM dd \nRDAT ee \nCDAT ff \nCOMP gg \nLOOP hh \nGRAD\nEND\nFFLD write ii \nCDAT\nCOMP jj "
#        self.temp_commands = self.commands



        self.font = "Helvetica 14 "
        self.output = ""
        self.FF = [("MM2",1),("MM3",2),("AMOEBA09",3)]
        self.QM = [("B3LYP",1),("PBE",2)]
        self.BAS = [("6-311G*",1),("6-31G",2)]
        self.CAL = [("Energy",1),("Force",2),("Hessian",3)]
        self.PRO = [("Gaussian",1),("PySCF",2)]
        self.LOS = [("Original",1),("Energy",2)]

        self.window()


    def window(self):
        win = Tk()
#        win.option_add("*Dialog.msg.font","Helvetica")
        self.menu(win)
        win.grid_rowconfigure(1, weight=1)
        win.grid_columnconfigure(0,weight=1)
        
        frame1 = Frame(win,width=160, height=45,relief=RIDGE,bg="red")
        frame1.grid(row=0,column=0)
        frame2 = Frame(win,relief=FLAT,width=160, height=45,bg="cyan")
        frame2.grid(row=2,column=0)
        
        frame3 = Frame(win,relief=RIDGE,width=160, height=45,bg="white")
        frame3.grid(row=0,column=1)

        
#        self.content(win)
        self.soft_frame(frame1)
        self.file_frame(frame2)
        self.weights_frame(frame3)

        win.mainloop()


    def menu(self,win):
        menu = Menu(win)
        win.config(menu=menu)
        filemenu = Menu(menu)
        # adding File menu
        menu.add_cascade(label="File", menu=filemenu)
        # options under Filemenu
        filemenu.add_command(label="New", command=self.newfile)
        filemenu.add_command(label="Open...", command= lambda inp = (("q2mm files","*.in"),("allfiles","*.*")): self.openfile(inp))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=win.quit)

        # Help menu
        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.about)

    def file_frame(self,win):
        return

    def weights_frame(self,win):
        cur_row = 0
        cur_col = 0
       
        
        weights = ["Weights"]
        for val in co.WEIGHTS:
            weights.append(val)
        for r in weights:
            Label(win,text=r, relief=FLAT,width=10).grid(row=cur_row,column=cur_col)
            cur_row += 1
        cur_row = 0
        cur_col += 1
        for val in co.WEIGHTS:
            fv = StringVar()
            fv.set(str(co.WEIGHTS[val]))
            cur_row += 1
            weight = Spinbox(win, from_=0, to=200, width=10, format="%10.2f", textvariable=fv)
            weight.grid(row=cur_row, column=cur_col)
            self.weights.append(weight)

    def soft_frame(self,win):
        col = 0
        row = 0
        col += 1
        row += 1
        cur_row = row
        options = ["Method","Quantum Mechanics","Files","Molecular Mechanics","Files","Options"]
        for r in options:
            Label(text=r, relief=RIDGE,width=20).grid(row=cur_row,column=0)
            cur_row += 1

        
        options = ["Charge","Energy","Bond","Angle","Torsion","Hessian","EigenMatrix"]
        var = []
        checks = []
        cur_row = row
        cur_col = 1
        for c in options:
            # Label
            temp_var = BooleanVar(0)
            temp_check = Checkbutton(win,text=c, width=20, variable=temp_var).grid(row=cur_row, column=cur_col)
            cur_col += 1
            var.append(temp_var)
            checks.append(temp_check)
        # QUANTUM MECHANICS
        cur_row = row+1
        cur_col = 1
        qm_options = ["Jaguar","Gaussian","Qchem"]
        i = 0
        checkboxs = []
        for c in options:
            cb = ttk.Combobox(win,values=qm_options)
            self.qm_soft[i] = cb
            #if var[i].get():
            cb.grid(row=cur_row, column=cur_col)
            cb.current(0)
            cur_col += 1
            i += 1
            checkboxs.append(cb)
        # FILE CALL
        row += 1
        cur_row = row+1
        cur_col = 1
        qm_files = []
        for c in options:
            qf=Button(win, text="Load", width=20,command=lambda inp=(("MAE files","*.mae"),("Gaussian output files","*.log"),("all files","*.*")): self.openfile(inp))
            qf.grid(row=cur_row, column=cur_col)
            cur_col += 1
            qm_files.append(qf)
        # MOLECULAR MECHANICS
        row += 1
        cur_row = row+1
        cur_col = 1
        mm_options = ["MacroModel","Tinker","Amber"]
        i = 0
        for c in options:
            cb = ttk.Combobox(win,values=mm_options)
            cb.grid(row=cur_row, column=cur_col)
            cb.current(0)
            self.mm_soft[i] = cb
            cur_col += 1
            i += 1
        # File Call
        row += 1
        cur_row = row+1
        cur_col = 1
        qm_files = []
        for c in options:
            qf=Button(win, text="Load", width=20,command=lambda inp=(("MAE files","*.mae"),("all files","*.*")): self.openfile(inp)).grid(row=cur_row, column=cur_col)
            cur_col += 1
            qm_files.append(qf)
        # Option for Charge
        row += 1
        cur_row = row+1
        cur_col = 1
        charge_options = ["Partial charge","Partial charge(no H)","Partial charge sum"] # q,qh,qa
        cb = ttk.Combobox(win,values=charge_options)
        cb.grid(row=cur_row, column=cur_col)
        cb.current(0)
        

        # Option for Energy
        cur_col += 1
        energy_options = ["Pre-opt energy","Average energy","Post-opt energy"] # e,ea,eao
        cb = ttk.Combobox(win,values=energy_options)
        cb.grid(row=cur_row, column=cur_col)
        cb.current(0)
        return

    def opt_frame(self,win):
        return


    def content(self,win):
        commands = self.commands
        col = 0
        row = 0
        cur_col = col
        cur_row = row
        Button(win, text="Working Directory", width=20, command = lambda inp=0: self.directory(inp)).grid(row=cur_row, column=cur_col)
        # working directory (DIR) aa

        row += 1
        cur_col = col
        cur_row = row
        Button(win, text="Open force field file", command=lambda inp=(("Macromodel ff parameters","*.fld"),("Tinker ff parameters","*.prm"),("Amber ff parameters","*.*"),("all file","*.*")):self.openfile(inp,0),width=20).grid(row=cur_row, column=cur_col)
        Button(win, text="Preview ff file", command=lambda inp=0: self.view_file(inp), width=20).grid(row=cur_row,column=cur_col+1)


#        commands += "DIR "+path
        # load forcefield file (FFLD read) bb
#        commands += "\nFFLD read "+path
        # load qparam file (PARM)
        
        
        # reference
        # calculation

        col += 1
        row += 1
        cur_row = row
        options = ["Method","Quantum Mechanics","Files","Molecular Mechanics","Files","Options"]
        for r in options:
            Label(text=r, relief=RIDGE,width=20).grid(row=cur_row,column=0)
            cur_row += 1

        
        options = ["Charge","Energy","Bond","Angle","Torsion","Hessian","EigenMatrix"]
        var = []
        checks = []
        cur_row = row
        cur_col = 1
        for c in options:
            # Label
            temp_var = BooleanVar(0)
            temp_check = Checkbutton(win,text=c, width=20, variable=temp_var).grid(row=cur_row, column=cur_col)
            cur_col += 1
            var.append(temp_var)
            checks.append(temp_check)
        # QUANTUM MECHANICS
        cur_row = row+1
        cur_col = 1
        qm_options = ["Jaguar","Gaussian","Qchem"]
        i = 0
        checkboxs = []
        for c in options:
            cb = Combobox(win,values=qm_options)
            self.qm_soft[i] = cb
            #if var[i].get():
            cb.grid(row=cur_row, column=cur_col)
            cb.current(0)
            cur_col += 1
            i += 1
            checkboxs.append(cb)
        # FILE CALL
        row += 1
        cur_row = row+1
        cur_col = 1
        qm_files = []
        for c in options:
            qf=Button(win, text="Load", width=20,command=lambda inp=(("MAE files","*.mae"),("Gaussian output files","*.log"),("all files","*.*")): self.openfile(inp))
            qf.grid(row=cur_row, column=cur_col)
            cur_col += 1
            qm_files.append(qf)
        # MOLECULAR MECHANICS
        row += 1
        cur_row = row+1
        cur_col = 1
        mm_options = ["MacroModel","Tinker","Amber"]
        i = 0
        for c in options:
            cb = Combobox(win,values=mm_options)
            cb.grid(row=cur_row, column=cur_col)
            cb.current(0)
            self.mm_soft[i] = cb
            cur_col += 1
            i += 1
        # File Call
        row += 1
        cur_row = row+1
        cur_col = 1
        qm_files = []
        for c in options:
            qf=Button(win, text="Load", width=20,command=lambda inp=(("MAE files","*.mae"),("all files","*.*")): self.openfile(inp)).grid(row=cur_row, column=cur_col)
            cur_col += 1
            qm_files.append(qf)
        # Option for Charge
        row += 1
        cur_row = row+1
        cur_col = 1
        charge_options = ["Partial charge","Partial charge(no H)","Partial charge sum"] # q,qh,qa
        cb = Combobox(win,values=charge_options)
        cb.grid(row=cur_row, column=cur_col)
        cb.current(0)
        

        # Option for Energy
        cur_col += 1
        energy_options = ["Pre-opt energy","Average energy","Post-opt energy"] # e,ea,eao
        cb = Combobox(win,values=energy_options)
        cb.grid(row=cur_row, column=cur_col)
        cb.current(0)

        # No Option for Bond, Angle, torsion
        # Weights
        self.weight_param(win,cur_row)

        # bond_options = []
        

        # eigenmatrix options for m changed by matching software


        # Convergence (LOOP)
        

        # name the final forcefield file(FFLD write)
        # run 
        
        row = cur_row + 2 

        self.checks = checks
        self.var = var
        Button(win, text="Debug", command=self.debug).grid(row=row,column=0)
        row += 1
        Button(win, text="Preview Q2MM Input File", command=self.preview).grid(row=row,column=0)
        row += 1
        Button(win, text="Save", command=self.debug,width=20).grid(row=row,column=0)
        Button(win, text="Run", command=self.debug,width=20).grid(row=row,column=1)
        Button(win, text="Quit", command=win.quit,width=20).grid(row=row,column=2)
        return

    # DEBUG
    def debug(self):
        for i in self.var:
            print(i.get())
#        for i in self.qm_soft:
#            print(i.get())
#        for i in self.mm_soft:
#            print(i.get())
        for i in range(len(self.var)):
            var = self.var[i].get()
            print(i,var)
            if var:
                print(self.qm_soft[i].get())
                print(self.mm_soft[i].get())
        for i in self.weights:
            print(i.get())

    def view_file(self,n):
        f = open(self.path[n],"r")
        messagebox.showinfo("Preview",f.read())
        #Text("1.0",f.read())

    def weight_param(self,win,row):
        cur_row = 0
        cur_col = 9
       
        
        weights = ["Weights"]
        for val in co.WEIGHTS:
            weights.append(val)
        for r in weights:
            Label(text=r, relief=FLAT,width=20).grid(row=cur_row,column=cur_col)
            cur_row += 1
        cur_row = 0
        cur_col += 1
        for val in co.WEIGHTS:
            fv = StringVar()
            fv.set(str(co.WEIGHTS[val]))
            cur_row += 1
            weight = Spinbox(win, from_=0, to=200, width=20, format="%10.2f", textvariable=fv)
            weight.grid(row=cur_row, column=cur_col)
            self.weights.append(weight)
            

        return

    def preview(self):
        aa = "" #DIR
        bb = "" #FFLD read
        cc = "" #WGHT
        dd = "" #PARM
        ee = "" #RDAT
        ff = "" #CDAT
        gg = "" #COMP
        hh = "" #LOOP
        ii = "" #FFLD write
        jj = "" #COMP

        if self.var[0].get():
            # read QM
            # read MM
            # read options
            # read files loaded
            # add lines to ee, ff
            0
        messagebox.showinfo("Q2MM Input File Preview", self.commands)
        print("COMMANDS:")
    # OS related functions

    def directory(self,n):
        self.cur_dir = filedialog.askdirectory(title="Select working directory")
    def newfile(self):
        print("Reset all variables")
    def openfile(self, filetype,n):
        path = filedialog.askopenfilename(initialdir=self.cur_dir, title="Select file",filetypes=filetype)
        self.path[n] = path
    def openfiles(self, filetype):
        paths = filedialog.askopenfilenames(initialdir=self.cur_dir, title="Select file",filetypes=filetype)
    def about(self):
        print("Q2MM is ")
q2mm_gui()
