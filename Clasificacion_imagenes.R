# En este ejemplo veremos como utilizar m??quinas de soporte vectorial para clasificar im??genes
source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")

# Cargaremos las librer??as necesarias
library(EBImage) # Procesamiento de im??genes
library(e1071) # M??quinas de soporte vectorial

# Fijemos el directorio de este trabajo
setwd("~/Dropbox/Cursos UdeC/Cursos pregrado/Data Mining/Maquinas de soporte vectorial")

# Carguemos la tabla de datos
olivetti_X <- read.csv('olivetti_X.csv', sep = ',', dec = '.', header = F)

# Miremos algunas im??genes
par(mfrow = c(8,8))
for (i in 1:64){
  img <- as.numeric(olivetti_X[i,])
  # Re-escalaremos las im??genes a 64*64
  img <- Image(img, dim=c(64, 64), colormode = "Grayscale")
  plot(img)
}

# Cargaremos el conjunto de etiquetas
oliveti_Y <- read.csv('olivetti_Y.csv', sep = ',', dec = '.', header = F)
oliveti_Y <- as.factor(oliveti_Y$V1)

# En el siguiente arreglo guardaremos las imagenes
rs_df <- data.frame()

# Las imagenes son muy grandes por lo de debemos reducir el n??mero de pixeles
for (i in 1:400){
  img <- as.numeric(olivetti_X[i,])
  # Re-escalaremos las im??genes a 64*64
  img <- Image(img, dim=c(64, 64), colormode = "Grayscale")
  # Re-escalamos las im??genes a 28x28 pixels
  img_resized <- resize(img, w = 28, h = 28)
  # Obtenemos la matriz asociada a cada imagen
  img_matrix <- img_resized@.Data
  # La transformamos en un vector
  img_vector <- as.vector(t(img_matrix))
  # La incluimos como fila en el arreglo
  rs_df <- rbind(rs_df, img_vector)
}

rs_df <- cbind(oliveti_Y, rs_df)
names(rs_df) <- c('Etiqueta', 1:784)

# Fijemos la semilla de modo de poder replicar los resultados
set.seed(100)

# Cambiemos al azar el orden de las filas
shuffled <- rs_df[sample(1:400),]

# Separemos los conjuntos de prueba y aprendizaje
train_28 <- shuffled[1:360, ]
test_28 <- shuffled[361:400, ]

# Entrenamos el modelo de m??quinas de soporte vectorial
model_svm <- svm(Etiqueta ~ . , train_28)

# Utilizamos el modelo para predecir la categor??a para el arreglo de prueba
pred <- predict(model_svm, test_28, type = "class")

# Matriz de confusi??n
Conf.Matrix <- table(test_28[,1], pred)
Conf.Matrix

# Algunos indicadores de precisi??n
P <- sum(diag(Conf.Matrix))/sum(Conf.Matrix)
print(paste('Obtenemos una precision del', P, '%'))
