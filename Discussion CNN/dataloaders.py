import numpy as np
from customdataset import customds
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import torch 

ola=np.loadtxt('result.csv',delimiter=',',skiprows=1) #Read data and skip header
atributos,espectros=ola[:,1:21],ola[:,21:] #Split based on column

print(atributos.shape,espectros.shape)

#State path for saving dataloaders
current_dir='C:\\Users\\Joaco\\Documents\\Universidad\\2024-1\\Estudios\\Deep learning\\Proyecto'
#dataloaders='Project_Dataloaders'
#dataloaders='Project_Normalized_Dataloaders'
dataloaders='Project_Globally_Normalized_Dataloaders_2'
union=os.path.join(current_dir,dataloaders)

#Split data into train, validation and test samples
#Setup hyperparameters
hyper={'train':.7,'test':.15,'val':.15,'batch_size':128}

SEED=42
# train is now 70% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(espectros, atributos, test_size=1-hyper['train'])

train_spectra_mean,train_attr_mean=np.mean(x_train),np.mean(y_train)
train_spectra_std,train_attr_std=np.std(x_train),np.std(y_train)

# test is now 15% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=hyper['test']/(hyper['test']+hyper['val']))

#Normalize train and validation data according to training mean and standard deviation.
x_train,x_val=(x_train-train_spectra_mean)/train_spectra_std,(x_val-train_spectra_mean)/train_spectra_std
y_train,y_val=(y_train-train_attr_mean)/train_attr_std, (y_val-train_attr_mean)/train_attr_std

if __name__=="__main__":
    print(f"Los valores de entrada a la red en los conjuntos de entrenamiento y validacion fueron normalizados utilizando una media de {train_spectra_mean} y una desviacion estandar de {train_spectra_std}")
    print(f"Los valores que buscamos predecir en los conjuntos de entrenamiento y validacion fueron normalizados utilizando una media de {train_attr_mean} y una desviacion estandar de {train_attr_std}")
#We're leaving testing data as it is.

x_train,x_val,x_test,y_train,y_val,y_test=torch.from_numpy(x_train).float(),torch.from_numpy(x_val).float(),torch.from_numpy(x_test).float(),torch.from_numpy(y_train).float(),torch.from_numpy(y_val).float(),torch.from_numpy(y_test).float()

if __name__=="__main__":
    print(f"x_train shape:{x_train.shape}, x_val shape: {x_val.shape}, x_test shape:{x_test.shape}\ny_train shape:{y_train.shape}, y_val shape: {y_val.shape}, y_test shape:{y_test.shape}")

train_dataset=customds(data=(x_train,y_train))
val_dataset=customds(data=(x_val,y_val),grad=False)
test_dataset=customds(data=(x_test,y_test),grad=False)

train_dataloader=DataLoader(train_dataset,batch_size=hyper['batch_size'],shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=1)
test_dataloader=DataLoader(test_dataset,batch_size=1)

if __name__=="__main__":
    torch.save(train_dataloader,os.path.join(union,'train_normalized.pth'))
    torch.save(val_dataloader,os.path.join(union,'val_normalized.pth'))
    torch.save(test_dataloader,os.path.join(union,'test_not_normalized.pth'))