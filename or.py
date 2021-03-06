"""
    author: Himanshu
    email:himaws72@gmail.com

"""



from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np 




def main(data,eta,epochs,filename,plotimage):
    
    df=pd.DataFrame(data)
    print(df)

    x,y=prepare_data(df)
    model =Perceptron(eta, epochs)
    model.fit(x,y)
    _=model.total_loss()


    save_model(model,filename)
    save_plot(df,plotimage,model)

if __name__== '__main__':
    OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]

    }
    ETA =0.3  # 0 and 1
    EPOCHS= 10
    main(data=OR ,eta=ETA,epochs=EPOCHS,filename="or.model",plotimage="or.png")
