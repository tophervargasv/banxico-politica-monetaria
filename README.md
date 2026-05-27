# Análisis de Política Monetaria Mexicana
## Series de Tiempo con Python

### ¿Qué hace este proyecto?
Este proyecto analiza la evolución de la inflación en México, su relación con la tasa de interés y el tipo de cambio, así como la respuesta del Banco de México con su tasa objetivo mediante la política monetaria. De igual forma se hace una proyección de inflación con modelos de series de tiempo.

### Datos
Se usan los datos del Sistema de Información Económica (SIE) del Banco de México mediante su API, desde enero de 2008 hasta la fecha más reciente disponible.

| Serie SIE | Indicador |
|---|---|
| SP30578 | Inflación general anual (INPC) |
| SP74662 | Inflación subyacente anual |
| SF61745 | Tasa objetivo Banxico |
| SF43718 | Tipo de cambio USD/MXN (FIX) |

### ¿Cómo ejecutarlo?
1. Abrir **Anaconda Prompt**
2. Activar el environment: `conda activate banxico`
3. Abrir Jupyter: `jupyter notebook`
4. Abrir el archivo y ejecutar cada celda en orden descendente

### Hallazgos principales
1. **Diagnóstico de datos** — La inflación mexicana 2008-2026 vivió tres shocks brutales: la crisis financiera 2008-2009, la pandemia en 2020, y el ciclo inflacionario post-pandemia 2021-2022.
2. **Causalidad** — La inflación Granger-causa a la tasa Banxico (lags 1-2, p<0.05), pero no al revés. El Banxico reacciona a la inflación con 1-2 meses de rezago — opera una regla de reacción, no de anticipación.
3. **Modelos** — ARIMA(0,1,1) proyecta línea plana, insuficiente. Prophet captura tendencia pero no shocks exógenos. El VAR con 3 lags es el más adecuado porque captura las interdependencias entre variables.
4. **Proyección** — El VAR anticipa inflación convergiendo hacia 4.25% y la tasa bajando a ~6% para finales de 2026 — ciclo de recortes sostenido pero inflación aún por encima de la meta del 3%.

### Dashboard en vivo
👉 https://banxico-politica-monetaria.streamlit.app/
