# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:02:56 2018

@author: sbece
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:15:07 2018

@author: elefra
"""

###ANÁLISIS DE DATOS CLIMATOLÓGICOS
##17 de Diciembre de 2018


##Integrantes:
##Sebastián Becerra
##Efraín Valencia


import netCDF4 as nc
import numpy as np
import scipy as sc
import datetime
import seaborn as sns

from Tkinter import *
from tkFileDialog   import askopenfilename    ##PARA ABRIR ARCHIVO
from tkMessageBox import * ##para abrir ventana con mensaje
import tkMessageBox
import webbrowser
from PIL import ImageTk, Image
import PIL.Image
import PIL.ImageTk
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits import basemap as bm
from tkSimpleDialog import *
import math
import string
import numpy as np
import pandas as pd
from scipy import stats  
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity ##para graficar kernels
import datetime
from scipy import stats

import xarray as xr
import statsmodels.api as sm  # para usar modelos arima
import netCDF4
from eofs.standard import Eof   

import warnings
import itertools


root = Tk()

def ayuda():
    window = Toplevel(root)
    window.geometry('500x300')
    window.wm_iconbitmap('Globe.ico')
    window.title("Ayuda")
    display = Label(window, text="Bienvidos a clima, el mejor programa para analizar datos climatológicos.\n\n No olvide que todos los datos deben estar en formato .nc y los rangos de \n longitud y latitud son (-180,180) y (-90,90) respectivamente.\n\n  La opción ver mapa le permite desplegar un mapa de presiones a nivel del\n mar para longitudes y latitudes escogidas. Además el ajuste de\n distribuciones le permite aplicar un test de normalidad a los datos. \n\n\n\n Para mayor información escríbanos a:\n \n sbecerra@udec.cl \n efrainvalencia@udec.cl",
                    font = ('Calibri',11), anchor = CENTER)
    display.pack()

class Clima(object):
    
    def __init__(self,archivo=[],data=[],dtime=[],muestra=[],latitud=[],longitud=[],anomalia=[],eof=[],pc=[]):
        self.archivo=archivo
        self.data=data
        self.dtime=dtime
        self.muestra=muestra
        self.latitud=latitud
        self.longitud=longitud
        self.anomalia=anomalia
        self.eof=eof
        self.pc=pc
    
    def abrir_archivo(self,name='', info=False, variables=False):
        name = askopenfilename() 
        self.archivo = nc.Dataset(name,'r')
        if info:
            print self.archivo
        if variables:
            print self.archivo.variables.keys()
        
    def ajustar_matriz(self,lat=[-90,90],lon=[0,360],time_i=[],time_f=[],level=[0,1000],m=False): #time_if(año,mes,dia,hora,minuto)
        #tiempo
        tiempo=self.archivo.variables['time']
        dtime=nc.num2date(tiempo[:],tiempo.units)
        c=0; time_ind=[]
        for i in range(0,len(dtime)):
            if dtime[i] == datetime.datetime(time_i[0],time_i[1],time_i[2],time_i[3]):
                time_ind.append(c)
            if dtime[i] == datetime.datetime(time_f[0],time_f[1],time_f[2],time_f[3]):
                time_ind.append(c+1)
            c+=1
        self.dtime=dtime[time_ind[0]:time_ind[1]]
        #latitud
        latitud=self.archivo.variables['lat'][:]
        lat_ajustado = (latitud>=lat[0]) * (latitud<=lat[1])
        self.latitud=latitud[lat_ajustado]
        #longitud
        longitud=self.archivo.variables['lon'][:]
        lon_ajustado = (longitud>=lon[0]) * (longitud<=lon[1])
        self.longitud=longitud[lon_ajustado]
        #nivel de presion
        if m:
            if 'level' in self.archivo.variables.keys():
                nivel=self.archivo.variables['level'][:]
                level_ajustado = (nivel>=level[0]) * (nivel<=level[1])
                matriz=self.archivo.variables[self.archivo.variables.keys()[len(self.\
                                              archivo.variables.keys())-1]][time_ind[0]\
                                              :time_ind[1],level_ajustado,lat_ajustado\
                                              ,lon_ajustado]
                (a,b,c,d)=np.shape(matriz)
                self.muestra=matriz.reshape((1,a*b*c*d))
            else:
                matriz=self.archivo.variables[self.archivo.variables.keys()[len(self.\
                                              archivo.variables.keys())-1]][time_ind[0]\
                                              :time_ind[1],lat_ajustado,lon_ajustado]
                (a,b,c)=np.shape(matriz)
                self.muestra=matriz.reshape((1,a*b*c))
        else:    
            if 'level' in self.archivo.variables.keys():
                nivel=self.archivo.variables['level'][:]
                level_ajustado = (nivel>=level[0]) * (nivel<=level[1])
                self.data=self.archivo.variables[self.archivo.variables.keys()[len(self.\
                                              archivo.variables.keys())-1]][time_ind[0]:\
                                              time_ind[1],level_ajustado,lat_ajustado,\
                                              lon_ajustado]
            else:
                self.data=self.archivo.variables[self.archivo.variables.keys()[len(self.archivo.\
                                              variables.keys())-1]][time_ind[0]:time_ind[1]\
                                              ,lat_ajustado,lon_ajustado]
         
    def anomalias(self,metodo='m',anom='normal'): #m:mes  e:estacion o estandarizado
        if metodo == 'm':
             c=0; 
             medias=np.ones((12,len(self.latitud),len(self.longitud)))
             stds=np.ones((12,len(self.latitud),len(self.longitud)))
             for i in range(0,12):
                 time_ind=[]
                 for j in range(0,len(self.dtime)):
                     d=self.dtime[j]
                     if getattr(d, 'month') == i+1:
                         time_ind.append(c)
                     c+=1
                 c=0
                 variable=self.data[time_ind,:,:]
                 medias[i,:,:]=np.mean(variable,axis=0) 
                 stds[i,:,:]=np.std(variable,axis=0) 
             a=np.ones(np.shape(self.data))
             if anom == 'normal':
                 for i in range(0,12):
                     a[i::12,:,:]=self.data[i::12,:,:]-medias[i,:,:]
                 self.anomalia=a
             elif anom == 'estandarizada':
                 for i in range(0,12):
                     a[i::12,:,:]=(self.data[i::12,:,:]-medias[i,:,:])/stds[i,:,:]
                 self.anomalia=a    
        elif metodo == 'e':
            c=0; 
            medias=np.ones((4,len(self.latitud),len(self.longitud)))
            stds=np.ones((4,len(self.latitud),len(self.longitud)))
            for i in range(0,4):
                time_ind=[]
                for j in range(0,len(self.dtime)):
                    d=self.dtime[j]
                    if getattr(d, 'month') == i+1 | getattr(d, 'month') == i+2 | getattr(d, 'month') == i+3 | getattr(d, 'month') == i+4:
                        time_ind.append(c)
                    c+=1
                c=0
                variable=self.data[time_ind,:,:]
                medias[i,:,:]=np.mean(variable,axis=0) 
                stds[i,:,:]=np.std(variable,axis=0)
            nuevo=np.ones((int(float(len(self.data[:,1,1]))/3.0),len(self.latitud),len(self.longitud)))
            for i in range(0,int(float(len(self.data[:,1,1]))/3.0)):
                nuevo[i,:,:]=np.mean(self.data[i:i+3,:,:])
            a=np.ones(np.shape(nuevo))
            if anom == 'normal':
                for i in range(0,4):
                    a[i::4,:,:]=nuevo[i::4,:,:]-medias[i,:,:]
                self.anomalia=a
            elif anom == 'estandarizada':
                for i in range(0,4):
                    a[i::4,:,:]=(nuevo[i::4,:,:]-medias[i,:,:])/stds[i,:,:]
                self.anomalia=a
                
    def EOF(self,metodo='correlacion',modo=1):
        window_eof = Toplevel(root, bg='brown') 
        window_eof.geometry('250x400')
        window_eof.wm_iconbitmap('Globe.ico')
        nombre1 = StringVar(window_eof, 'm')
        nombre2 = StringVar(window_eof, 'estandarizada')
        nombre3 = IntVar(window_eof, 1)
        nombre4 = StringVar(window_eof, 'correlacion')
        label1 = Label(window_eof, text="Método de anomalía:", width=15).place(x=10, y=100)
        label2 = Label(window_eof, text="Tipo de anomalía:", width=15).place(x=10, y=150)
        label3 = Label(window_eof, text="Modo:", width=15).place(x=10, y=200)
        label4 = Label(window_eof, text="Método de la EOF:", width=15).place(x=10, y=250)
        entry_box1 = Entry(window_eof, textvariable=nombre1, width=15).place(x=140, y=100)
        entry_box2 = Entry(window_eof, textvariable=nombre2, width=15).place(x=140, y=150)
        entry_box3 = Entry(window_eof, textvariable=nombre3, width=15).place(x=140, y=200)
        entry_box3 = Entry(window_eof, textvariable=nombre4, width=15).place(x=140, y=250)
        self.anomalias(metodo=nombre1.get(),anom=nombre2.get())
        lat=self.latitud[::-1]
        coslat = np.cos(np.deg2rad(lat))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = Eof(self.anomalia, weights=wgts) ##SE NECESITA LA LIBRERIA eofs y tengo que escribir from eofs.standard import Eof          
        #eof=np.ones((3,len(self.latitud),len(self.longitud)))
        if nombre4.get() == 'correlacion':
            self.eof = solver.eofsAsCorrelation(neofs=3)
            titulo='EOF por Correlacion - Modo '+ str(modo)
            barra='Correlacion'
        elif nombre4.get() == 'covarianza':
            self.eof = solver.eofsAsCovariance(neofs=3)
            titulo='EOF por Covarianza - Modo '+ str(modo)
            barra='Covarianza'
            
        def desplegar_eof1():
            self.pc = solver.pcs(npcs=nombre3.get(), pcscaling=1)
            plt.figure(figsize=(10,4))
            plt.plot_date(self.dtime, self.pc, fmt='-')
            plt.grid()
            plt.ylabel('var.units')
            #plt.title('%s at Lon=%.2f, Lat=%.2f' % (vname, loni, lati))
            plt.savefig('pc.png')
            ven_graf_serie = Toplevel(window_eof)
            ven_graf_serie.minsize(400, 400)
            ven_graf_serie.wm_iconbitmap('Globe.ico')
            ven_graf_serie.title("Componente principal modo "+str(nombre3.get()))
            im = PIL.Image.open("pc.png")
            photo = PIL.ImageTk.PhotoImage(im)
            label = Label(ven_graf_serie, image=photo)
            label.image = photo  # keep a reference!
            label.pack() 
            
        despliegue1 = Button(window_eof, text="Desplegar Anomalia", command=desplegar_eof1).place(x=60, y=300)
    
        m = bm.Basemap(llcrnrlon = self.longitud[0],llcrnrlat = self.latitud[len(self.latitud)-1],\
            urcrnrlon = self.longitud[len(self.longitud)-1],urcrnrlat = self.latitud[0],projection = 'merc')
        lat=self.latitud
        lon=self.longitud
        lon, lat = np.meshgrid(lon, lat)
        x, y = m(lon, lat)
        print np.shape(self.eof)
        
        def desplegar_eof2():
            fig = plt.figure(figsize=(8,6))
            fig.add_axes([0.05,0.05,0.9,0.85])
            csf = m.contourf(x,y,self.eof[nombre3.get()-1,:,:].squeeze(),np.arange(-1,1,0.1),cmap='jet')
            m.drawcoastlines(linewidth=1.25, color='black')
            m.drawparallels(np.arange(-180,180,20),labels=[1,0,0,0],color='gray')
            m.drawmeridians(np.arange(-180,180,20),labels=[0,0,0,1],color='gray')
            cbar = m.colorbar(csf,location='right',size='3%') #location puede ser top, left or right
            cbar.set_label(barra)
            plt.title(titulo)
        #plt.show()
            plt.savefig('eof.png')
            ven_graf_serie = Toplevel(window_eof)
            ven_graf_serie.minsize(400, 400)
            ven_graf_serie.wm_iconbitmap('Globe.ico')
            ven_graf_serie.title("Campo espacial de EOF")
            im = PIL.Image.open("eof.png")
            photo = PIL.ImageTk.PhotoImage(im)
            label = Label(ven_graf_serie, image=photo)
            label.image = photo  # keep a reference!
            label.pack() 
            
        despliegue2 = Button(window_eof, text="Campo espacial EOF", command=desplegar_eof2).place(x=60, y=350)
    
            
    def resumen_datos(self):
        window = Toplevel(root) 
        window.geometry('400x200')
        window.wm_iconbitmap('Globe.ico')
        window.configure(background='lightblue')
        
        def descriptivo():
            datoss = np.asarray(self.muestra)
            descr = Toplevel(window) 
            descr.geometry('300x150')
            descr.configure(background='white')
            Label(descr, text="El promedio es: "+str(np.mean(datoss)), font = ('Calibri',11), bg='white').grid(row=2)
            Label(descr, text="La desviación estándar es: "+str(np.std(datoss)), font = ('Calibri',11), bg='white').grid(row=6)
            Label(descr, text="El mínimo valor es: "+str(np.amin(datoss)), font = ('Calibri',11), bg='white').grid(row=8)
            Label(descr, text="El máximo valor es: "+str(np.amax(datoss)), font = ('Calibri',11), bg='white').grid(row=12)         
       
        def localizacion():
            media=np.asarray(self.muestra)
            media=np.mean(media)
            q25=np.percentile(self.muestra,25)
            q50=np.percentile(self.muestra,50)
            q75=np.percentile(self.muestra,75)
            trimean=(q25+2*q50+q75)/4
            local = Toplevel(window, bg='white') 
            local.geometry('350x100')
            Label(local, text="La media es: "+str(media), font = ('Calibri',11), bg='white').grid(row=2)
            Label(local, text="El rango intercuartílico es: "+str(trimean), font = ('Calibri',11), bg='white').grid(row=6)
#        print 'No resistente - Media: ', media
#        print 'Resistente - Trimean: ', trimean 

        
        def simetria():
            sk = sc.stats.skew(self.muestra)
            q25=np.percentile(self.muestra,25)
            q50=np.percentile(self.muestra,50)
            q75=np.percentile(self.muestra,75)
            IQR=q75-q25
            YK=(q25-2*q50+q75)/IQR
            sim = Toplevel(window, bg='white') 
            sim.geometry('350x100')
            Label(sim, text="La simetría o sesgo es: "+str(sk), font = ('Calibri',11), bg='white').grid(row=2)
            Label(sim, text="Índice de Yule-Kendall: "+str(IQR), font = ('Calibri',11), bg='white').grid(row=6)
#        print 'No resistente - Skewness: ', sk
#        print 'Resistente - : ', YK #buscar  nombre de esto
#        
        despliegue1 = Button(window, text="Estadísticos descriptivos", command=descriptivo).place(x=50, y=50)
        despliegue2 = Button(window, text="Localidad", command=localizacion).place(x=50, y=100)
        despliegue3 = Button(window, text="Simetría", command=simetria).place(x=50, y=150)

    
    def usar_por_defecto(self): ##precipitación, humedad relativa, temperatura superficial del mar, temeperatura del aire, velocidad del viento, dirección del viento, contenido de agua precipitable
        window = Toplevel(root, bg='red') 
        window.geometry('250x400')
        window.wm_iconbitmap('Globe.ico')
        
        def ventanita():
            ventana = Toplevel(window, bg='lightblue') 
            ventana.geometry('400x400')
            ventana.wm_iconbitmap('Globe.ico')
            nombre1 = DoubleVar(ventana, 20); nombre2 = DoubleVar(ventana, 70)
            nombre3 = DoubleVar(ventana, 20); nombre4 = DoubleVar(ventana, 50)
            year1 = IntVar(ventana, 2000); mes1 = IntVar(ventana, 1); dia1 = IntVar(ventana, 1); hora1 = IntVar(ventana, 0)
            year2 = IntVar(ventana, 2010); mes2 = IntVar(ventana, 1); dia2 = IntVar(ventana, 1); hora2 = IntVar(ventana, 0)
            label1 = Label(ventana, text="Longitud oeste:").place(x=10, y=50)
            label2 = Label(ventana, text="Longitud este:").place(x=10, y=100)
            label3 = Label(ventana, text="Latitud sur:").place(x=10, y=150)
            label4 = Label(ventana, text="Latitud norte:").place(x=10, y=200)
            label5 = Label(ventana, text="Tiempo inicial:").place(x=10, y=250)
            label6 = Label(ventana, text="Tiempo final:").place(x=10, y=300)
            entry_box1 = Entry(ventana, textvariable=nombre1).place(x=100, y=50)
            entry_box2 = Entry(ventana, textvariable=nombre2).place(x=100, y=100)
            entry_box3 = Entry(ventana, textvariable=nombre3).place(x=100, y=150)
            entry_box4 = Entry(ventana, textvariable=nombre4).place(x=100, y=200)
            entry_box5_1 = Entry(ventana, textvariable=year1, width=10).place(x=100, y=250)
            entry_box5_2 = Entry(ventana, textvariable=mes1, width=10).place(x=150, y=250)
            entry_box5_3 = Entry(ventana, textvariable=dia1, width=10).place(x=200, y=250)
            entry_box5_4 = Entry(ventana, textvariable=hora1, width=10).place(x=250, y=250)
            entry_box6_1 = Entry(ventana, textvariable=year2, width=10).place(x=100, y=300)
            entry_box6_2 = Entry(ventana, textvariable=mes2, width=10).place(x=150, y=300)
            entry_box6_3 = Entry(ventana, textvariable=dia2, width=10).place(x=200, y=300)
            entry_box6_4 = Entry(ventana, textvariable=hora2, width=10).place(x=250, y=300)
            
            def guardar_datos():
                var1 = nombre1.get()
                var2 = nombre2.get()
                var3 = nombre3.get()
                var4 = nombre4.get()
                var5 = year1.get()
                var6 = mes1.get()
                var7 = dia1.get()
                var8 = hora1.get()
                var9 = year2.get()
                var10 = mes2.get()
                var11 = dia2.get()
                var12 = hora2.get()
                self.ajustar_matriz(lat=[var3,var4],lon=[var1,var2],time_i=[var5,var6,var7,var8],time_f=[var9,var10,var11,var12])
            ## ajustar_matriz se guarda en la variable data
#            def guardar_datos():
#                self.data = 
#            
            despliegue8 = Button(ventana, text="Guardar datos", command=guardar_datos).place(x=150, y=350)
#        
        def temperatura_sst():
            self.archivo = nc.Dataset('sst.mnmean.nc','r')
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        def presion():
            self.archivo = nc.Dataset('slp.mon.mean.nc','r')
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        def precipitacion():
            self.archivo = nc.Dataset('precip.mon.mean.nc','r')
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        def humedad_relativa():
            self.archivo = nc.Dataset('rhum.mon.mean.nc','r')
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        def temperatura_aire():
            self.archivo = nc.Dataset('air.mon.mean.nc','r')
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        def vel_viento():
            self.archivo = nc.Dataset('wspd.mon.mean.nc','r')  
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        def contenido():
            self.archivo = nc.Dataset('pr_wtr.eatm.mon.mean.nc','r')  
            self.titulo = 'Temperatura superficial del mar'
            ventanita()
            
        
        despliegue1 = Button(window, text="Temperatura superficial del mar", command=temperatura_sst).place(x=25, y=50)
        despliegue2 = Button(window, text="Presion", command=presion).place(x=25, y=100)
        despliegue3 = Button(window, text="Precipitacion", command=precipitacion).place(x=25, y=150)
        despliegue4 = Button(window, text="Humedad relativa", command=humedad_relativa).place(x=25, y=200)
        despliegue5 = Button(window, text="Temperatura del aire", command=temperatura_aire).place(x=25, y=250)
        despliegue6 = Button(window, text="Velocidad del viento", command=vel_viento).place(x=25, y=300)
        despliegue7 = Button(window, text="Contenido de agua precipitable", command=contenido).place(x=25, y=350)
        
        
    def test_shapiro(self):
    # normality test according to shapiro wilks test
        stat, p = shapiro(self.data)
        window = Toplevel(root, bg='lightblue') 
        window.configure(background='white')
        window.geometry('700x300')
        window.wm_iconbitmap('Globe.ico')
        Label(window, text='Test de normalidad de Shapiro Wilks:', font = ('Calibri',14), bg='white').grid(row=0)
        Label(window, text='Statistics=%.3f, p=%.3f' % (stat, p), font = ('Calibri',12), bg='white').grid(row=4)
    # interpret
        alpha = 0.05
        if p > alpha:
            Label(window, text='Los datos siguen una distribución normal (no existe evidencia significativa para rechazar H0)', font = ('Calibri',12), bg='white').grid(row=8)
        else:
            Label(window, text='Los datos no siguen una distribución normal (existe evidencia significativa para rechazar H0)', font = ('Calibri',12), bg='white').grid(row=8)
    
    def caja(self):
        window_caja = Toplevel(root, bg='lightblue') 
        window_caja.geometry('200x200')
        window_caja.configure(background='brown')
        window_caja.wm_iconbitmap('Globe.ico')

        def figura():
            plt.figure()
            plt.boxplot(self.muestra)
            plt.savefig("boxplot.png")
            ven_caja = Toplevel(window_caja, bg='lightblue')
            ven_caja.minsize(400, 400)
            ven_caja.title("Boxplot")
            im = PIL.Image.open("boxplot.png")
            photo = PIL.ImageTk.PhotoImage(im)
            label = Label(ven_caja, image=photo)
            label.image = photo  # keep a reference!
            label.pack()
        despliegue1 = Button(window_caja, width=10,  text="Desplegar", command=figura).place(x=50, y=100)
    
    
    def mapa(self,proj='merc'):
        ventana = Toplevel(root, bg='lightblue') 
        ventana.geometry('400x300')
        ventana.wm_iconbitmap('Globe.ico')
        year1 = IntVar(); mes1 = IntVar(); dia1 = IntVar(); hora1 = IntVar()
        label1 = Label(ventana, text="Tiempo:").place(x=10, y=50)
        entry_box1 = Entry(ventana, textvariable=year1, width=10).place(x=100, y=50)
        entry_box1 = Entry(ventana, textvariable=mes1, width=10).place(x=150, y=50)
        entry_box1 = Entry(ventana, textvariable=dia1, width=10).place(x=200, y=50)
        entry_box1 = Entry(ventana, textvariable=hora1, width=10).place(x=250, y=50)
        
        
        
        def desplegar_mapa():
            va1 = year1.get()
            va2 = mes1.get()
            va3 = dia1.get()
            va4 = hora1.get()
            tiempo = [va1, va2, va3, va4]
            c=0; 
            for i in range(0,len(self.dtime)):
                if self.dtime[i] == datetime.datetime(tiempo[0],tiempo[1],tiempo[2],tiempo[3]):
                    time_ind=c
                c+=1
            datos_mapa=self.data[time_ind,:,:]
            m = bm.Basemap(llcrnrlon = self.longitud[0],llcrnrlat = self.latitud[len(self.latitud)-1],\
            urcrnrlon = self.longitud[len(self.longitud)-1],urcrnrlat = self.latitud[0],projection = proj)
        #llcrnrlon: longitud esquina inferior izquierda
        #llcrnrlat: latitud esquina inferior izquierda
        #urcrnrlon: longitud esquina superior derecha
        #urcrnrlat: latitud esquina superior derecha
        ## Encontramos los valores x,y para el grid de la proyección del mapa.
            lat=self.latitud
            lon=self.longitud
            lon, lat = np.meshgrid(lon, lat)
            x, y = m(lon, lat)
            titulo=self.archivo.variables[self.archivo.variables.keys()[len(self.\
                                              archivo.variables.keys())-1]]
            fondo=[np.min(self.data),np.max(self.data),(np.max(self.data)-np.min(self.data))/12]
            fig = plt.figure(figsize=(8,10))
            fig.add_axes([0.05,0.05,0.9,0.85])
            csf = m.contourf(x,y,datos_mapa.squeeze(),np.arange(fondo[0],fondo[1],fondo[2]),cmap='jet')
            m.drawcoastlines(linewidth=1.25, color='black')
            m.drawparallels(np.arange(-180,180,20),labels=[1,0,0,0],color='gray')
            m.drawmeridians(np.arange(-180,180,20),labels=[0,0,0,1],color='gray')
            cbar = m.colorbar(csf,location='right',size='3%') #location puede ser top, left or right
            cbar.set_label(titulo.units)
            plt.title(titulo.long_name+' '+str(va3)+'/'+str(va2)+'/'+str(va1))
            #plt.show()
            fig.savefig('mapa.png')
            ven_mapa = Toplevel(ventana, bg='lightblue')
            ven_mapa.minsize(600, 600)
            ven_mapa.title("Mapa")
#        #img = ImageTk.PhotoImage(Image.open("foto.png"))
#        #l=Label(image=img)
#        #l.pack()
            im = PIL.Image.open("mapa.png")
            photo = PIL.ImageTk.PhotoImage(im)
            label = Label(ven_mapa, image=photo)
            label.image = photo  # keep a reference!
            label.pack()       
        despliegue = Button(ventana, text="Desplegar mapa", command=desplegar_mapa).place(x=100, y=190)
        
    
    def histograma(self):
        window_histo = Toplevel(root, bg='lightblue')
        window_histo.minsize(200, 200)
        window_histo.wm_iconbitmap('Globe.ico')
        
        def desplegar():
            hist = plt.hist(self.muestra, bins='auto')
            plt.title(self.titulo)
            #plt.xlabel("Days")
            plt.ylabel("Frequency")
            plt.savefig("histograma.png")
            ven_histo = Toplevel(window_histo)
            ven_histo.minsize(400, 400)
            ven_histo.title("Histograma")
            im_histo = PIL.Image.open("histograma.png")
            photo_histo = PIL.ImageTk.PhotoImage(im_histo)
            label = Label(ven_histo, image=photo_histo)
            label.image = photo_histo  # keep a reference!
            label.pack() 
        despliegue = Button(window_histo, text="Desplegar histograma", command=desplegar).place(x=50, y=100)

    
    def ajuste_distribucion(self):
        window_dist = Toplevel(root, bg='lightblue')
        window_dist.minsize(400, 400)
        window_dist.wm_iconbitmap('Globe.ico')
        def desplegar_normal():
        # plot normed histogram
            plt.hist(self.muestra, normed=True)

        # find minimum and maximum of xticks, so we know
        # where we should compute theoretical distribution
            xt = plt.xticks()[0]  
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(self.data))

        # lets try the normal distribution first
            m, s = stats.norm.fit(self.data) # get mean and standard deviation  
            pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
            plt.plot(lnspc, pdf_g, label="Norm") # plot it
            plt.savefig("histograma_normal.png")
            ven1_dist = Toplevel(window_dist, bg='white')
            ven1_dist.minsize(400, 400)
            ven1_dist.title("Ajuste de distribucion normal a histograma")
            im1 = PIL.Image.open("histograma_normal.png")
            photo1 = PIL.ImageTk.PhotoImage(im1)
            label1 = Label(ven1_dist, image=photo1)
            label1.image = photo1  # keep a reference!
            label1.pack() 
        def desplegar_gamma():
        # plot normed histogram
            plt.hist(self.muestra, normed=True)

        # find minimum and maximum of xticks, so we know
        # where we should compute theoretical distribution
            xt = plt.xticks()[0]  
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(self.data))
         # exactly same as above
            ag,bg,cg = stats.gamma.fit(self.data)  
            pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)  
            plt.plot(lnspc, pdf_gamma, label="Gamma")
            plt.savefig("histograma_gamma.png")
            ven2_dist = Toplevel(window_dist, bg='white')
            ven2_dist.minsize(200, 200)
            ven2_dist.title("Ajuste de distribucion gamma a histograma")
            im2 = PIL.Image.open("histograma_gamma.png")
            photo2 = PIL.ImageTk.PhotoImage(im2)
            label2 = Label(ven2_dist, image=photo2)
            label2.image = photo2  # keep a reference!
            label2.pack()
        def desplegar_beta():
            plt.hist(self.muestra, normed=True)

        # find minimum and maximum of xticks, so we know
        # where we should compute theoretical distribution
            xt = plt.xticks()[0]  
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(self.data))
         # exactly same as above
            ab,bb,cb,db = stats.beta.fit(ser)  
            pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)  
            plt.plot(lnspc, pdf_beta, label="Beta")
            plt.savefig("histograma_beta.png")
            ven3_dist = Toplevel(window_dist, bg='white')
            ven3_dist.minsize(200, 200)
            ven3_dist.title("Ajuste de distribucion beta a histograma")
            im3 = PIL.Image.open("histograma_beta.png")
            photo3 = PIL.ImageTk.PhotoImage(im3)
            label3 = Label(ven3_dist, image=photo3)
            label3.image = photo3  # keep a reference!
            label3.pack()
        despliegue1 = Button(window_dist, text="Normal", command=desplegar_normal).place(x=100, y=200)
        despliegue2 = Button(window_dist, text="Gamma", command=desplegar_gamma).place(x=160, y=200)
        despliegue3 = Button(window_dist, text="Beta", command=desplegar_beta).place(x=230, y=200)
    
    
    def grafico_serie(self):
        lat = self.latitud
        lon = self.longitud
        jd = self.dtime
        window_graf_serie = Toplevel(root, bg='lightblue') 
        window_graf_serie.geometry('400x400')
        window_graf_serie.wm_iconbitmap('Globe.ico')
        nombre1 = DoubleVar()
        nombre2 = DoubleVar()
        label1 = Label(window_graf_serie, text="Latitud:").place(x=10, y=100)
        label2 = Label(window_graf_serie, text="Longitud:").place(x=10, y=150)
        entry_box1 = Entry(window_graf_serie, textvariable=nombre1).place(x=100, y=100)
        entry_box2 = Entry(window_graf_serie, textvariable=nombre2).place(x=100, y=150)
        def anual():
            loni = nombre1.get()
            lati = nombre2.get()
            lat_ajustado = (lat==lati)
            lati=lat[lat_ajustado]
            lon_ajustado = (lon==loni)
            loni=lon[lon_ajustado]
            v = self.archivo.variables[self.archivo.variables.keys()[len(self.\
                                              archivo.variables.keys())-1]]
            vname=v.long_name
            h = self.data[:,lat_ajustado,lon_ajustado]
            plt.figure(figsize=(10,4))
            plt.plot_date(jd,h,fmt='-')
            plt.grid()
            plt.ylabel(v.units)
            plt.title('%s at Lon=%.2f, Lat=%.2f' % (vname, loni, lati))
            plt.savefig('anual.png')
            ven_graf_serie = Toplevel(window_graf_serie)
            ven_graf_serie.minsize(400, 300)
            ven_graf_serie.wm_iconbitmap('Globe.ico')
            ven_graf_serie.title("Serie de tiempo anual")
            im = PIL.Image.open("anual.png")
            photo = PIL.ImageTk.PhotoImage(im)
            label = Label(ven_graf_serie, image=photo)
            label.image = photo  # keep a reference!
            label.pack() 
        despliegue1 = Button(window_graf_serie, text="Desplegar serie", command=anual).place(x=50, y=200)    
        
        
    def analisis_serie(self):
        nuevo = self.archivo
        lat = nuevo.variables['lat'][:]
        lon = nuevo.variables['lon'][:]
        times = nuevo.variables['time']
        jd = netCDF4.num2date(times[:],times.units)
        window_serie = Toplevel(root, bg='lightblue') 
        window_serie.geometry('400x400')
        window_serie.wm_iconbitmap('Globe.ico')
        nombre1 = DoubleVar()
        nombre2 = DoubleVar()
        label1 = Label(window_serie, text="Latitud:").place(x=10, y=100)
        label2 = Label(window_serie, text="Longitud:").place(x=10, y=150)
        entry_box1 = Entry(window_serie, textvariable=nombre1).place(x=100, y=100)
        entry_box2 = Entry(window_serie, textvariable=nombre2).place(x=100, y=150)
        def resultados():
            
            loni = nombre1.get()
            lati = nombre2.get()
            vname = 'Sea level pressure'
            var = nuevo.variables['slp']
            h = var[:, lati, loni]
            
            # Define the p, d and q parameters to take any value between 0 and 2
            p = d = q = range(0, 2)

            # Generate all different combinations of p, q and q triplets
            pdq = list(itertools.product(p, d, q))
            
            # Generate all different combinations of seasonal p, q and q triplets
            seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

            print('Examples of parameter combinations for Seasonal ARIMA...')
            print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
            print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
            print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
            print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
            
            warnings.filterwarnings("ignore") # specify to ignore warning messages

            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(h,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                    
                        results = mod.fit()
                    
                        resultados = 'ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic)

                        #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                        print results.AIC
                    except:
                        continue
            ###Se ajusta el modelo con el menor AIC
            mod = sm.tsa.statespace.SARIMAX(h,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

            results = mod.fit()
            
            print(results.summary().tables[1])
            results.plot_diagnostics(figsize=(8, 8))
            #plt.show()
            plt.savefig('modelo.png')
            ven_serie = Toplevel(window_serie, bg='lightblue')
            ven_serie.minsize(600, 600)
            ven_serie.title("Resultados ajuste modelo ARIMA")
            im = PIL.Image.open("modelo.png")
            photo = PIL.ImageTk.PhotoImage(im)
            label = Label(ven_serie, image=photo)
            label.image = photo  # keep a reference!
            label.pack()       
        despliegue = Button(window_serie, text="Desplegar resultados", command=resultados).place(x=100, y=300) 
        
    
    
prueba=Clima()



root.minsize(500, 500)
root.title("Clima")
root.wm_iconbitmap('Globe.ico')

##How to create menu
menu = Menu(root)
root.config(menu=menu)

###How to create a submenu
subMenu = Menu(menu) ##A menu inside the menu
menu.add_cascade(label = "Archivo", menu = subMenu) ##Create a file button
subMenu.add_command(label="Abrir archivo",command=prueba.abrir_archivo)
subMenu.add_command(label="Usar archivo por defecto",command=prueba.usar_por_defecto)
subMenu.add_command(label="Ayuda",command=ayuda)

subMenu.add_separator()
subMenu.add_command(label = "Salir", command=root.destroy)

editMenu = Menu(menu)
menu.add_cascade(label="Ver", menu=editMenu)  ##A dropout functionality
editMenu.add_cascade(label="Mapa", command=prueba.mapa)
editMenu.add_cascade(label="EOF con anomalías", command=prueba.EOF)
editMenu.add_cascade(label="Histograma", command=prueba.histograma) #Gráfico de probabilidades
editMenu.add_cascade(label="Boxplot", command=prueba.caja)
editMenu.add_cascade(label="Gráfico de probabilidades", command=prueba.ajuste_distribucion)
editMenu.add_cascade(label="Serie de tiempo", command=prueba.grafico_serie)


#
editMenu = Menu(menu)
menu.add_cascade(label="Estadísticas", menu=editMenu)  ##A dropout functionality
editMenu.add_cascade(label="Tabla resumen de datos", command=prueba.resumen_datos)
editMenu.add_cascade(label="Ajuste de distribuciones", command=prueba.test_shapiro)
editMenu.add_cascade(label="Modelos de series temporales", command=prueba.analisis_serie)
#editMenu.add_cascade(label="Funciones empíricas ortogonales", command=create_window)


## Status bar
status = Label(root, text="Cargando...", bd=1, relief=SUNKEN, anchor=W) ##relief is how you want your border
status.pack(side=BOTTOM, fill=X)


root.mainloop()