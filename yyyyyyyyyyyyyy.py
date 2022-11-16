from tkinter import X


while(1):
    try:
        x = input()
        if int(x[0]) == 0:
            break
        a = 0
        yy =  [int(i) for i in x.split()]
        print(yy)
        for i in yy[1:]:
            a = a + i
        print(a)
    except:
        break