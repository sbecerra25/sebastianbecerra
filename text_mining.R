##Fijamos el dorectorio de trabajo
setwd("~/Dropbox/Cursos UdeC/Cursos pregrado/Data Mining/Text Mining")
cname <- file.path(".", "trump_mining")

##Verificamos que la ruta tenga el archivo requerido
dir(cname)

#install.packages("tm", dependencies=TRUE)
library(tm)
corpus <- Corpus(DirSource(cname))
corpus
summary(corpus)

#inspect(corpus[1])

##Transformaciones interesantes
# Convertimos todo en min??sculas
corpus <- tm_map(corpus, tolower)
# Quitamos los n??meros
corpus <- tm_map(corpus, removeNumbers)
# Quitamos los signos de puntuaci??n
corpus <- tm_map(corpus, removePunctuation)
# Se eliminan varias palabras comunes del ingl??s
corpus <- tm_map(corpus, removeWords, stopwords("english"))
#corpus <- tm_map(corpus, removeWords, stopwords("spanish"))

# Quitamos los espacios que han salido con tanto retoque
corpus <- tm_map(corpus, stripWhitespace)
# Nos aseguramos que el corpus es texto plano
corpus <- tm_map(corpus, PlainTextDocument)

#Vamos ahora a cargar el paquete ???SnowballC??? para hacer Stemming.

#install.packages("SnowballC", dependencies = TRUE)
library(SnowballC)
corpus <- tm_map(corpus, stemDocument)
# Quitamos los espacios que han salido con tanto retoque
corpus <- tm_map(corpus, stripWhitespace)

inspect(corpus[1])

# Matriz de t??rminos
dtm <- DocumentTermMatrix(corpus)
dtm

### Exploramos nuestra Matriz de terminos
freq <- colSums(as.matrix(dtm))
length(freq)
ord <- order(freq)

# Lista terminos menos frecuentes
freq[head(ord)]

# Lista terminos m??s frecuentes
freq[tail(ord)]


## Distribuci??n de las frecuencias
head(table(freq), 15)
freq

tail(table(freq), 15)
freq
names(freq[freq>100])

## Eliminar t??rminos infrecuentes
dim(dtm)
dtms <- removeSparseTerms(dtm, .1)
dim(dtms)
inspect(dtms)


##Nube de palabras o Wordcloud
#install.packages("wordcloud", dependencies = TRUE)
library(wordcloud)
palette <- brewer.pal(9,"BuGn")[-(1:5)]
wordcloud(names(freq), freq, min.freq=50, rot.per=0.2, scale=c(3, .1), colors=palette,vfont=c("serif","plain"))
wordcloud

## Identificar terminos m??s frecuentes y sus relaciones
freq <- colSums(as.matrix(dtms))
findFreqTerms(dtm, lowfreq=25)
findFreqTerms(dtm, lowfreq=20)

##Dibujamos las frecuencias
freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
head(freq, 14)
wf <- data.frame(word=names(freq), freq=freq)
head(wf)
#Dibujamos las palabras con m??s de 15 frecuencia
#install.packages("ggplot2", dependencies = TRUE)
library(ggplot2)
p <- ggplot(subset(wf, freq>3), aes(word, freq))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
p
