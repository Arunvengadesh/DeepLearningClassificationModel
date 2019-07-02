import random
def GenerateData(size):
    
    Attr = []
    classi = []

    for i in range(0,size):
        Shape = round(random.uniform(1,10),1)
        if (Shape > 5):
            
            
            Length1 = round(random.uniform(1,50),1)
            Length2 = Length1
            Brth1 = round(random.uniform(50,100),1)
            Brth2 = Brth1
            clas = [0.0,0.1]
            A = [Length1,Brth1,Length2,Brth2]
            Attr.append(A)
            classi.append(clas)
           
            
        else:
            
            
            
            Side1 = round(random.uniform(1,100),1)
            Side2 = Side3 = Side4 = Side1
            clas = [1.0,0.0]
            A = [Side1,Side2,Side3,Side4]
            Attr.append(A)
            classi.append(clas)
            

    return Attr,classi            
