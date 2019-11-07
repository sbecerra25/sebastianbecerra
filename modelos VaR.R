install.packages("PerformanceAnalytics")
install.packages("quantmod")
library(PerformanceAnalytics)
library(quantmod)
library(rugarch)

install.packages("car")
library(car)
install.packages("FinTS")

library(FinTS)



# cargar los datos 

getSymbols("MSFT", from ="2000-01-03", to = "2018-09-28")
colnames(MSFT)
start(MSFT)
end(MSFT)
head(MSFT)


# grafico de los precios 

plot(MSFT)

# log-returns 

MSFT = MSFT[, "MSFT.Adjusted", drop=F]
MSFT.ret = CalculateReturns(MSFT, method="log")

rtn<-dailyReturn(MSFT)
head(rtn)
head(MSFT.ret)

# remove first NA observation
MSFT.ret = MSFT.ret[-1,]
rtn= rtn[-1,]

all.equal(as.numeric(MSFT.ret), as.numeric(rtn)) 

colnames(MSFT.ret) ="MSFT"

# grafco de los retornos 

plot(MSFT.ret)

# grafco de los retornos con cuadrados y valor absoluto de los retornos

dataToPlot = cbind(MSFT.ret, MSFT.ret^2, abs(MSFT.ret))
colnames(dataToPlot) = c("Returns", "Returns^2", "abs(Returns)")
plot.zoo(dataToPlot, main="MSFT Daily Returns", col="blue")

# Estimate GARCH(1,1)
#
# specify GARCH(1,1) model with only constant in mean equation
garch11.spec = ugarchspec(variance.model = list(garchOrder=c(1,1)), ##con errores gausianos
                          mean.model = list(armaOrder=c(0,0)))

MSFT.garch11.fit = ugarchfit(spec=garch11.spec, data=MSFT.ret)
MSFT.garch11.fit

acf(residuals(MSFT.garch11.fit))

# residuals: e(t)
plot.ts(residuals(MSFT.garch11.fit), ylab="e(t)", col="blue")
abline(h=0)

# sigma(t) = conditional volatility
plot.ts(sigma(MSFT.garch11.fit), ylab="sigma(t)", col="blue")

plot(MSFT.garch11.fit,)

var <- quantile(MSFT.garch11.fit, p=0.05)
plot(var)
#
# Prediccion
#

MSFT.garch11.fcst = ugarchforecast(MSFT.garch11.fit, n.ahead=100)
names(MSFT.garch11.fcst@forecast)

MSFT.garch11.fcst
plot(MSFT.garch11.fcst)

par(mfrow=c(2,1))
plot(MSFT.garch11.fcst, which=1)
plot(MSFT.garch11.fcst, which=3)
par(mfrow=c(1,1))

#
# calcular el pron??stico de la varianza del d??a h = suma de los pron??sticos anticipados de la varianzadel d??a h 
#

MSFT.fcst.var.hDay = cumsum(MSFT.garch11.fcst@forecast$sigmaFor^2) ###MSFT.fcst.df$sigma^2)
MSFT.fcst.vol.hDay = sqrt(MSFT.fcst.var.hDay)


#
# Predicciones del VaR
#

names(MSFT.garch11.fcst)
# h step-ahead conditional normal GARCH(1,1) VaR
media<-MSFT.garch11.fcst@forecast$seriesFor
volatilidad<-MSFT.garch11.fcst@forecast$sigmaFor

VaR.95.garch11 = media[1] + volatilidad*qnorm(0.05)
VaR.95.garch11

ts.plot(VaR.95.garch11)

# compute 20-day vol forecast from fitted GARCH(1,1)
sigma.20day = sqrt(sum(volatilidad[1:20]^2))
VaR.95.garch11.20day = 20*media[1] + sigma.20day*qnorm(0.05)
VaR.95.garch11.20day
ts.plot(VaR.95.garch11)

#
# backtesting para  el calculo del VaR
# normal VaR y HS 
#

# Ventana de prueba y estimaci??n 

n.obs = nrow(MSFT.ret)
w.e = 2000
w.t = n.obs - w.e
alpha = 0.95
?VaR

#
# Calcular el VaR para la muestra de prueba y registrar los excesos 
#

backTestVaR <- function(x, p = 0.95) {
  normal.VaR = as.numeric(VaR(x, p=p, method="gaussian"))
  historical.VaR = as.numeric(VaR(x, p=p, method="historical"))
  ans = c(normal.VaR, historical.VaR)
  names(ans) = c("Normal", "HS")
  return(ans)
}

VaR.results = rollapply(as.zoo(MSFT.ret), width=w.e, 
                        FUN = backTestVaR, p=0.95, by.column = FALSE,
                        align = "right")

chart.TimeSeries(merge(MSFT.ret, VaR.results), legend.loc="topright")

?VaR
############

violations.mat = matrix(0, 2, 5)
rownames(violations.mat) = c("Normal", "HS")
colnames(violations.mat) = c("En1", "n1", "1-alpha", "Percent", "VR")
violations.mat[, "En1"] = (1-alpha)*w.t
violations.mat[, "1-alpha"] = 1 - alpha

# Muestra las violaciones del VaR caso  Normal 

normalVaR.violations = as.zoo(MSFT.ret[index(VaR.results), ]) < VaR.results[, "Normal"]
violation.dates = index(normalVaR.violations[which(normalVaR.violations)])

# plot de violaciones 

plot(as.zoo(MSFT.ret[index(VaR.results),]), col="blue", ylab="Return")
abline(h=0)
lines(VaR.results[, "Normal"], col="black", lwd=2)
lines(as.zoo(MSFT.ret[violation.dates,]), type="p", pch=16, col="red", lwd=2)

for(i in colnames(VaR.results)) {
  VaR.violations = as.zoo(MSFT.ret[index(VaR.results), ]) < VaR.results[, i]
  violations.mat[i, "n1"] = sum(VaR.violations)
  violations.mat[i, "Percent"] = sum(VaR.violations)/w.t
  violations.mat[i, "VR"] = violations.mat[i, "n1"]/violations.mat[i, "En1"]
}

violations.mat


#### Caso Normal
?VaRTest
VaR.test = VaRTest(1-alpha, 
                   actual=coredata(MSFT.ret[index(VaR.results),]), 
                   VaR=coredata(VaR.results[,"Normal"]))
names(VaR.test)

# LR test for correct number of exceedances
VaR.test[1:7]

# LR tests for independence of exceedances
VaR.test[8:12]

### Caso HS

VaR.test = VaRTest(1-alpha, 
                   actual=coredata(MSFT.ret[index(VaR.results),]), 
                   VaR=coredata(VaR.results[,"HS"]))
names(VaR.test)

# LR test for correct number of exceedances
VaR.test[1:7]

# LR tests for independence of exceedances
VaR.test[8:12]


# Usando rugarch package para calcular las violaciones del VaR
?ugarchroll
MSFT.garch11.roll = ugarchroll(garch11.spec, MSFT.ret, n.ahead=1,
                               forecast.length = w.t,
                               refit.every=20, refit.window="moving",
                               calculate.VaR=TRUE, VaR.alpha=0.01)
class(MSFT.garch11.roll)

plot(MSFT.garch11.roll)
# VaR plot
plot(MSFT.garch11.roll, which=4)
# Coef plot`
plot(MSFT.garch11.roll, which=5)
# show backtesting report
?report
report.msft = report(MSFT.garch11.roll, type="VaR")


xx = VaRTest(alpha=0.01, actual=MSFT.garch11.roll@forecast$VaR[,2], 
             VaR=MSFT.garch11.roll@forecast$VaR[,1])



