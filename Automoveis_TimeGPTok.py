import numpy as np
import pandas as pd

from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series
from utilsforecast.losses import mae, mse, mape, rmse, smape
from nixtla import NixtlaClient
import streamlit as st

nixtla_client = NixtlaClient('nixak-vgekzclAmNcdPsAOSfaQ7WcDXWgGjU6X5peo92QoKCUKG4wDFq3AO1yU6GXStQRyGPHVw21Zao78DknY'
    # api_key = 'my_api_key_provided_by_nixtla'
)

# Configurações gerais da página
st.set_page_config(
    page_title="TimeGPT Predizendo Licenciamentos para Segmento de Automóveis",
    page_icon="🚗",
    layout="wide"
)

col1, col2 = st.columns([1,1], gap= 'large')

with col1:
   
    st.header('TimeGPT: Predizendo Licenciamentos para Segmento de Automóveis', divider= 'green')
    st.header("_TimeGPT_ is :blue[cool] :sunglasses:")
    st.markdown('''O TimeGPT é uma ferramenta avançada de previsão que utiliza técnicas de machine learning e análise de séries temporais para prever licenciamentos no segmento de automóveis. Com base em dados históricos e variáveis econômicas, o TimeGPT oferece previsões precisas e confiáveis, auxiliando empresas e órgãos reguladores a tomar decisões informadas. A ferramenta é capaz de identificar padrões e tendências, proporcionando insights valiosos para o planejamento estratégico e a otimização de recursos no setor automotivo.
''')

with col2:    
       
    st.image('https://ph-files.imgix.net/de28c977-5ecb-4f2b-a1fb-e744d6181f3f.png?auto=format&fit=crop', width=800)

nixtla_client.validate_api_key()

df = pd.read_excel(r'C:\Tablets\lic_Veículos_1.xlsx')
df = df.rename(columns={"Mês": "ds"})

col1, col2 = st.columns([1,2], gap='large')

df['ds'] = pd.to_datetime(df['ds'])
series_cols = ['Automóveis']
df_long = df.melt(id_vars='ds', value_vars=series_cols, var_name='unique_id', value_name='y')

st.header('Dataframe e Gráfico - Licenciamento Automóveis', divider='red')

col1, col2 = st.columns([1,2.5], gap='large')

with col1:
    st.write('Dataframe Padrão TimeGPT')
    st.dataframe(df_long, height=500)

with col2:
    st.write('Gráfico Licenciamentos Automóveis')
    st.image("C:\Tablets\data_auto.png", width= 1500)

st.header('Identificação de Anomalias', divider='gray')

col1, col2 = st.columns([1.2,2], gap= 'large')

with col1:
    st.markdown('''TimeGPT é uma ferramenta avançada de previsão e detecção de anomalias em séries temporais. A detecção de anomalias é uma tarefa crucial que identifica pontos fora do comportamento normal da série, sendo essencial em diversas aplicações, como cibersegurança e monitoramento de equipamentos.
Aqui estão os principais pontos sobre a detecção de anomalias com TimeGPT:
- **Identificação de Pontos Anômalos**: TimeGPT utiliza intervalos de confiança para determinar se um ponto é anômalo. Se um ponto cai fora desse intervalo, ele é considerado uma anomalia.
''')
    st.write('Dataframe Anomalias Licenciamentos Automóveis')    
    anomalies_df = nixtla_client.detect_anomalies(df_long, freq='ME', level=90)
    st.dataframe(anomalies_df, height=1000)
    nixtla_client.plot(df_long, anomalies_df)

with col2:
    st.write('Gráfico Anomalias Licenciamentos Automóveis')
    st.markdown('''A imagem parece ser uma série temporal com o comportamento da variável "Target [y]" ao longo do tempo (em segundos), comparando os dados reais com uma previsão gerada por um modelo **TimeGPT**.

---

### 1. **Componentes principais do gráfico**  
- **Linha azul escura (`y`)**: Representa os valores reais da variável ao longo do tempo.  
- **Linha rosa (`TimeGPT`)**: Representa os valores preditos pelo modelo TimeGPT.  
- **Faixa rosa clara (`TimeGPT_level_90`)**: Intervalo de confiança de 90% para a previsão.  
- **Pontos vermelhos (`TimeGPT_anomalies_level_90`)**: Pontos identificados como anomalias pelo modelo.

---

### 2. **Comportamento observado**  
- **Períodos de queda e recuperação**:  
   - Houve uma **queda acentuada** nos valores reais em torno de **2020**, seguida de uma recuperação gradual.  
   - A queda pode estar associada a algum evento específico, a Covid 19.  
- **Previsões do modelo TimeGPT**:  
   - A previsão segue razoavelmente próxima dos valores reais, exceto em pontos onde ocorrem **anomalias**.
   - A faixa rosa (intervalo de confiança) se **alarga** em períodos de maior incerteza.  
- **Anomalias**:  
   - Os **pontos vermelhos** indicam discrepâncias significativas entre o valor real e o previsto.  
   - As anomalias são mais evidentes em períodos de queda extrema e recuperação em 2020.

---

### 3. **Conclusões**  
- O modelo TimeGPT capturou bem o padrão geral da série temporal, inclusive suas tendências e variações.  
- As **anomalias** coincidem com períodos de comportamento atípico, indicando momentos de **quebra de padrão** ou **eventos inesperados**.  
- O **intervalo de confiança** se expande em pontos de maior variabilidade ou incerteza, refletindo um comportamento esperado para previsões.

---

### 4. **Próximos passos sugeridos**  
- **Analisar as causas das anomalias**: Investigue os eventos externos ou internos que possam ter causado os desvios significativos em 2020.  
- **Refinar o modelo**: Pode-se ajustar o modelo ou usar técnicas adicionais para tratar períodos anômalos.  
- **Avaliar a faixa de incerteza**: A ampliação do intervalo sugere que a variabilidade aumenta em certas regiões, o que pode indicar sazonalidade ou ruídos.
''')
   
    st.image('C:\\Tablets\\anomalias.png', width=1300)


nixtla_client.plot(df_long, time_col='ds', target_col='y')

timegpt_fcst_df = nixtla_client.forecast(df=df_long, h=36, freq='MS', 
                                         time_col='ds', 
                                         target_col='y',
                                         add_history=True, 
                                         level=[80, 90, 95])

st.header('Forecast TimeGPT', divider='blue')

col1, col2 = st.columns([2,2], gap='large')

with col1:
    st.write("Dataframe Forecast Licenciamentos Automóveis TimeGPT - Intervalo de Confiança 80, 90 e 95%")
    st.dataframe(timegpt_fcst_df, height=800)

with col2:

    st.write('Gráfico Forecast Licenciamentos Automóveis TimeGPT')
    st.image('C:\\Tablets\\fcsttimegpt.png', width=1000)
    st.markdown('''### Análise do Gráfico

O gráfico representa a série temporal da variável **"Target [y]"** ao longo do tempo e as previsões geradas pelo modelo **TimeGPT**. Os componentes visuais fornecem insights importantes sobre o comportamento dos dados e o desempenho do modelo.

---

### Componentes principais do gráfico:
1. **Linha azul escura (`y`)**: Representa os valores **reais** da variável ao longo do tempo.
2. **Linha rosa (`TimeGPT`)**: Representa as previsões feitas pelo modelo **TimeGPT**.
3. **Faixas rosa claras**:  
   - **TimeGPT_level_80**: Intervalo de confiança de 80%.  
   - **TimeGPT_level_90**: Intervalo de confiança de 90%.  
   - **TimeGPT_level_95**: Intervalo de confiança de 95%.  
   Quanto mais claro o tom da faixa, maior o nível de confiança (maior a incerteza).

---

### Comportamento Observado:
1. **Tendência Geral**:  
   - A série temporal real (**linha azul**) apresenta um comportamento com variações ao longo do tempo, com uma **queda brusca** em torno de **2020**, possivelmente devido a algum evento externo, seguida de uma recuperação gradual.

2. **Previsões (TimeGPT)**:  
   - O modelo **TimeGPT** acompanha bem os valores reais após a queda, mostrando que ele capturou adequadamente a tendência e sazonalidade presentes nos dados.  
   - A linha rosa está próxima da linha azul real, indicando boa precisão na previsão.

3. **Intervalos de Confiança**:  
   - As **faixas rosa claras** representam a incerteza nas previsões.  
   - Durante o período de **2025 em diante**, as faixas de incerteza aumentam gradativamente, refletindo a **maior incerteza** associada a previsões de longo prazo.  
   - O intervalo de confiança de **95% (faixa mais clara)** é visivelmente mais amplo, o que é esperado em previsões estatísticas.

4. **Impacto do Período de Queda**:  
   - A queda acentuada em **2020** pode ter influenciado o modelo, mas ele ainda conseguiu projetar valores consistentes no período de recuperação.

---

### Conclusões:
1. O modelo **TimeGPT** está **desempenhando bem**, pois suas previsões acompanham a série real com um bom nível de proximidade.
2. O aumento na **incerteza das previsões** ao longo do tempo é normal e deve ser considerado na interpretação dos resultados.  
3. A **queda em 2020** deve ser analisada com mais detalhes, pois pode indicar a influência de eventos externos ou anomalias que podem distorcer a série histórica.

---

### Recomendações:
1. **Analisar eventos externos**: Investigue o que ocorreu no período de 2020 que provocou a queda abrupta.
2. **Avaliar a qualidade das previsões**: Calcule métricas como **RMSE**, **MAE**, ou **MAPE** para quantificar a precisão do modelo.
3. **Cenários futuros**: Utilize o intervalo de confiança para considerar diferentes cenários (otimista e pessimista) em planejamentos futuros.
''')


nixtla_client.plot(df_long, timegpt_fcst_df, 
                   time_col='ds', 
                   target_col='y',
                   level=[80, 90, 95])


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.header('Métricas de Previsão - Mean Squared Error, Mean Absolute Error, Mean Absolute Percent Error, Room Mean Squared Error, Bias, Bias_Rated, Simetric Mape', divider= 'green')

col1, col2= st.columns([2,2], gap='large')

metricas = pd.read_excel(r'C:\Tablets\auto_timegpt_metricas.xlsx')
metricas = metricas.rename(columns={"Mês": "ds"})
   
# Calcular os erros
y_true = metricas['Automóveis']
y_pred = metricas['y_pred']

# Métricas
mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
rmse = np.sqrt(mse)  # Root Mean Squared Error

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# SMAPE (Symmetric Mean Absolute Percentage Error)
smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

# BIAS (Mean Bias Deviation)
bias = np.mean(y_pred - y_true)

# Bias Rated
bias_rated = (bias / np.mean(y_true)) * 100

# Exibir os resultados
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Bias_rated: {bias_rated:.2f}%")

with col1:

# Exibindo as métricas com Streamlit
    st.write("### Métricas de Avaliação do Modelo")  # Título

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
    st.write(f"**Symmetric Mean Absolute Percentage Error (SMAPE):** {smape:.2f}%")
    st.write(f"Bias: {bias:.2f}")
    st.write(f"**Bias Rated:** {bias_rated:.2f}%")
    st.markdown('''### Análise das Métricas de Avaliação do Modelo

Os valores fornecidos representam as **métricas de erro e viés** do modelo. Vamos analisá-los individualmente e interpretar sua qualidade para o desempenho do modelo.
---

### **1. Mean Squared Error (MSE): 1.166.382.887,23**
- **Definição**: O MSE mede o erro quadrático médio entre os valores reais e preditos.  
- **Interpretação**:  
   - O valor elevado do **MSE** indica que existem erros significativos entre as previsões e os valores reais.  
   - Por ser quadrático, essa métrica é muito sensível a **outliers** (erros muito grandes), o que pode estar inflando o valor.

---

### **2. Mean Absolute Error (MAE): 23.750,46**
- **Definição**: Mede o erro absoluto médio entre os valores reais e preditos, mantendo a unidade dos dados.  
- **Interpretação**:  
   - O **MAE de 23.750,46** significa que, em média, o modelo apresenta um erro de **23.750 unidades**.  
   - Embora seja mais fácil de interpretar que o MSE, esse valor ainda pode ser considerado alto dependendo da escala dos dados.

---

### **3. Root Mean Squared Error (RMSE): 34.152,35**
- **Definição**: O RMSE é a raiz quadrada do MSE, trazendo o erro para a mesma unidade dos dados.  
- **Interpretação**:  
   - O **RMSE de 34.152,35** representa a **magnitude média do erro**.  
   - Como o RMSE dá maior peso a grandes erros, ele é sempre maior ou igual ao MAE.  
   - A diferença entre RMSE e MAE sugere que existem **outliers** ou previsões muito distantes dos valores reais.
''')
    
with col2:
    st.markdown('''### **4. Mean Absolute Percentage Error (MAPE): 20,57%**
- **Definição**: O MAPE mede o erro percentual médio entre os valores reais e previstos.  
- **Interpretação**:  
   - Um **MAPE de 20,57%** indica que, em média, as previsões estão **20,57% fora** dos valores reais.  
   - Esse valor está na faixa aceitável, mas sugere que o modelo pode ser melhorado.  
   - Modelos com MAPE abaixo de **10%** são considerados muito precisos, entre **10% e 20%** são razoáveis, e acima de **20%** podem indicar dificuldades.        
                
### **5. Symmetric Mean Absolute Percentage Error (SMAPE): 15,59%**
- **Definição**: O SMAPE é uma versão ajustada do MAPE que considera tanto os valores reais quanto os previstos.  
- **Interpretação**:  
   - O **SMAPE de 15,59%** confirma que o erro percentual é moderado.  
   - Valores próximos a **15%** indicam que o modelo tem um desempenho razoável, mas há espaço para melhorias.

---

### **6. Bias: 9.330,23**
- **Definição**: O **Bias** representa a média das diferenças entre os valores previstos e reais. Ele indica se o modelo tende a **superestimar** ou **subestimar** os valores.  
- **Interpretação**:  
   - O valor positivo de **9.330,23** sugere que o modelo apresenta um **viés positivo**, ou seja, as previsões são, em média, **9.330 unidades maiores** que os valores reais.  
   - Esse viés é significativo e deve ser analisado mais profundamente.

---

### **7. Bias Rated: 6,02%**
- **Definição**: O **Bias Rated** representa o viés percentual, indicando a proporção média em que as previsões estão superestimadas ou subestimadas.  
- **Interpretação**:  
   - O **Bias Rated de 6,02%** significa que, em média, o modelo **superestima os valores reais em 6,02%**.  
   - Embora o valor não seja excessivamente alto, é um sinal de que o modelo apresenta tendência de **superestimação constante**.

### **Conclusão Geral**
Com base nas métricas fornecidas:
- O **erro absoluto (MAE: 23.750,46)** e o **erro quadrático (RMSE: 34.152,35)** estão altos, sugerindo que o modelo possui erros significativos.  
- O **erro percentual** (MAPE: 20,57% e SMAPE: 15,59%) mostra que o modelo tem um desempenho **razoável**, mas longe de ser excelente.  
- O **viés positivo** (Bias: 9.330,23 e Bias Rated: 6,02%) indica que o modelo tem uma tendência de **superestimar** os valores reais.
''')

timegpt_fcst_finetune_mae_df = nixtla_client.forecast(
    df=df_long, 
    h=36, 
    finetune_steps=60,
    finetune_loss='mae',   # Set your desired loss function
    finetune_depth=1,
    time_col='ds', 
    target_col='y',
    level=[80, 90, 95],
    add_history=True
)


timegpt_fcst_finetune_mae_df.to_excel('C:\Tablets/autotimegptfinetun1.xlsx', index=False)
nixtla_client.plot(df_long, timegpt_fcst_finetune_mae_df, 
                   time_col='ds', 
                   target_col='y',
                   level=[80, 90, 95])

import pandas as pd
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
import numpy as np

# Divisão do conjunto de dados em treino e teste
train = df_long[:-36]
test = df_long[-36:]

# Exibir as partes de treino e teste
print("Train Data:")
print(train.head())

print("\nTest Data:")
print(test.head())

# Visualização dos dados
plot_series(train[['unique_id', 'ds', 'y']], 
            forecasts_df=test[['unique_id', 'ds', 'y']].rename(columns={'y': 'test'}))

# Lista de perdas a serem testadas
losses = ['default', 'mae', 'mse', 'rmse', 'mape', 'smape']

# Criar uma cópia do conjunto de teste para inserir previsões
test = test.copy()

# Loop para realizar previsões com diferentes funções de perda
for loss in losses:
    preds_df = nixtla_client.forecast(
        df=train, 
        h=36, 
        finetune_steps=60,    
        finetune_loss=loss,
        time_col='ds', 
        target_col='y'
    )
    preds = preds_df['TimeGPT'].values  # Obter as previsões
    test.loc[:, f'TimeGPT_{loss}'] = preds  # Armazenar previsões no conjunto de teste

# Dicionário de funções de perda
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

loss_fct_dict = {
    "mae": lambda test, c1, c2: mae(test['y'], test[c2]),
    "mse": lambda test, c1, c2: mse(test['y'], test[c2]),
    "rmse": lambda test, c1, c2: rmse(test['y'], test[c2]),
    "mape": lambda test, c1, c2: mape(test['y'], test[c2]),
    "smape": lambda test, c1, c2: smape(test['y'], test[c2])
}

# Avaliação e cálculo da melhoria percentual
pct_improv = []

for loss in losses[1:]:
    eval_default = loss_fct_dict[loss](test, 'y', 'TimeGPT_default')
    eval_loss = loss_fct_dict[loss](test, 'y', f'TimeGPT_{loss}')
    
    pct_diff = (eval_default - eval_loss) / eval_default * 100
    pct_improv.append(round(pct_diff, 2))

# Construção do DataFrame final com as melhorias
data = {
    'mae': pct_improv[0],
    'mse': pct_improv[1],
    'rmse': pct_improv[2],
    'mape': pct_improv[3],
    'smape': pct_improv[4]
}

metrics_df = pd.DataFrame(data, index=['Metric Improvement (%)'])

# Exibir as métricas de melhoria
print("\nMetrics Improvement DataFrame:")
print(metrics_df)

st.header('Aprimorando a Modelagem por Meio do TimeGPT FineTuning_steps = 60, FineTuning_Mae e FineTuning_depths = 1', divider= 'violet')

col1, col2 = st.columns([1.5,4], gap='large')
    
with col1: 
    st.write('"_TimeGPT FineTuning - Aprimorando Previsões_ é :blue[top] :sunglasses:"')
    st.dataframe(metrics_df)

    st.markdown('''### **Resumo Explicativo das Vantagens do FineTuning**

O código apresentado tem como objetivo **avaliar e comparar o desempenho do modelo TimeGPT** ao usar diferentes **funções de perda (loss functions)** durante o ajuste fino (**fine-tuning**). Abaixo estão as principais vantagens e benefícios desse código:

---

### **1. Teste de Diferentes Funções de Perda**
- O código permite avaliar o impacto de diferentes funções de perda (**MAE, MSE, RMSE, MAPE, SMAPE**) no desempenho do modelo.
- Isso possibilita **identificar a melhor função de perda** para um determinado conjunto de dados e contexto.
- A abordagem é flexível, pois utiliza uma lista de funções que pode ser facilmente expandida.

**Benefício**: Melhoria da precisão das previsões ao escolher a função de perda mais adequada.

---

### **2. Automação do Processo de Avaliação**
- A estrutura de loop automatiza a **realização das previsões** com todas as funções de perda definidas.
- Métricas de desempenho são calculadas automaticamente para cada previsão e armazenadas de forma organizada.

**Benefício**: Ganho de **eficiência e produtividade**, eliminando a necessidade de ajustar manualmente o modelo várias vezes.

---

### **3. Comparação com um Padrão (`default`)**
- O desempenho das previsões ajustadas é comparado diretamente com a previsão padrão (`TimeGPT_default`).
- Isso permite calcular a **melhoria percentual** obtida ao usar diferentes funções de perda.

**Benefício**: Facilita a **quantificação dos ganhos de desempenho** e torna a comparação mais objetiva.

---

### **4. Avaliação com Múltiplas Métricas**
- O código calcula várias métricas de erro:
  - **MAE**: Mede o erro absoluto médio, fácil de interpretar.
  - **MSE** e **RMSE**: Penalizam erros maiores, sendo úteis para previsões sensíveis a grandes desvios.
  - **MAPE** e **SMAPE**: Fornecem erros percentuais, facilitando a comparação entre diferentes escalas de dados.
- O uso de múltiplas métricas garante uma avaliação mais completa e balanceada do desempenho.

**Benefício**: Garante uma **análise robusta** do modelo ao não depender de uma única métrica.

---

### **5. Estrutura Clara e Organizada**
- As previsões são armazenadas em colunas específicas no conjunto de teste, facilitando a organização dos dados.
- O cálculo de melhorias percentuais é armazenado em um **DataFrame** final, o que facilita a interpretação dos resultados.

**Benefício**: **Visualização fácil** e organizada dos resultados, ideal para análise e apresentação.

---

### **6. Adaptação a Diferentes Contextos**
- O código é facilmente adaptável a outros modelos ou conjuntos de dados.
- Novas funções de perda ou métricas podem ser adicionadas ao código sem grandes modificações.

**Benefício**: **Versatilidade** para aplicações em diferentes contextos e necessidades.

---

### **Conclusão**
O código oferece uma abordagem automatizada, flexível e robusta para **avaliar o impacto de diferentes funções de perda** em um modelo de previsão. Com ele, é possível **identificar a melhor estratégia** de ajuste fino para melhorar a precisão das previsões, economizando tempo e facilitando a análise dos resultados. Essa prática é extremamente útil em problemas de séries temporais e ajuda a tomar decisões mais embasadas no desempenho do modelo.
''')



import pandas as pd
from nixtla import NixtlaClient
from utilsforecast.losses import mae, mse
from utilsforecast.evaluation import evaluate

pd.options.display.float_format = '{:,.2f}'.format

depths = [1, 2, 3, 4, 5]

test = test.copy()

for depth in depths:
    preds_df = nixtla_client.forecast(
    df=train, 
    h=36, 
    finetune_steps=5,
    finetune_depth=depth,
    time_col='ds', 
    target_col='y')

    preds = preds_df['TimeGPT'].values

    test.loc[:,f'TimeGPT_depth{depth}'] = preds

    test['unique_id'] = 0

evaluation = evaluate(test, metrics=[mae, mse], time_col="ds", target_col="y")
with col2:
    st.write('"_TimeGPT FineTuning - Aprimorando Previsões_ é :blue[top] :sunglasses:"')
    st.dataframe(evaluation)
    st.markdown('''### **Análise do DataFrame de Melhoria das Métricas**

O DataFrame **Metrics Improvement** apresenta os resultados da **melhoria percentual** de diversas métricas de erro ao comparar previsões feitas com diferentes funções de perda em relação à previsão padrão (`TimeGPT_default`). As métricas avaliadas são **MAE**, **MSE**, **RMSE**, **MAPE** e **SMAPE**.

A seguir, analisaremos os resultados de cada métrica:

---

### **1. MAE (Mean Absolute Error)**
- **Melhoria percentual**: **2.89%**
- O **MAE** é uma métrica que mede o erro absoluto médio das previsões. Uma melhoria de **2.89%** indica que a função de perda utilizada **reduziu ligeiramente o erro absoluto médio** em relação ao modelo padrão. Embora a melhoria não seja muito grande, qualquer redução no erro absoluto é geralmente desejável, especialmente quando se trata de prever valores com unidades reais.

**Interpretação**: A função de perda testada **contribuiu de forma modesta para melhorar a precisão** do modelo em termos de erro absoluto.

---

### **2. MSE (Mean Squared Error)**
- **Melhoria percentual**: **9.1%**
- O **MSE** penaliza erros maiores mais severamente do que o MAE, tornando-o útil para identificar modelos que cometem grandes erros. Uma melhoria de **9.1%** sugere que a função de perda foi eficaz em **reduzir os erros quadráticos**, indicando que a função de perda testada ajudou a diminuir os grandes desvios, o que pode ser crucial em muitos problemas de previsão, especialmente em séries temporais.

**Interpretação**: A função de perda testada teve um impacto **mais significativo** na redução de grandes erros.

---

### **3. RMSE (Root Mean Squared Error)**
- **Melhoria percentual**: **3.37%**
- O **RMSE** é a raiz quadrada do MSE e mede o erro médio, dando mais peso aos grandes erros. Uma melhoria de **3.37%** significa que, embora tenha havido uma redução no RMSE, o impacto foi mais **modesto** em comparação com a melhoria observada no MSE. O RMSE é particularmente útil para entender o erro em unidades comparáveis aos valores de saída.

**Interpretação**: A função de perda ajudou a **reduzir levemente** o erro médio ponderado, mas a melhoria foi menor que no MSE, o que é esperado dado que o RMSE tem uma relação mais direta com a escala dos valores.

---

### **4. MAPE (Mean Absolute Percentage Error)**
- **Melhoria percentual**: **59.44%**
- O **MAPE** mede o erro médio percentual, que é útil para avaliar o desempenho de modelos em termos relativos. Uma melhoria de **59.44%** é bastante significativa, indicando que a função de perda testada **reduziu drasticamente o erro percentual**. Isso sugere que, ao usar essa função de perda, as previsões do modelo ficaram muito mais **precisas em relação aos valores reais**.

**Interpretação**: A função de perda testada proporcionou uma **grande melhoria** na precisão percentual das previsões, o que é extremamente importante em muitos cenários onde a precisão relativa é crucial.

---

### **5. SMAPE (Symmetric Mean Absolute Percentage Error)**
- **Melhoria percentual**: **8.53%**
- O **SMAPE** é uma métrica similar ao MAPE, mas é mais equilibrada no tratamento de erros relativos, tratando igualmente os desvios positivos e negativos. A melhoria de **8.53%** indica que a função de perda testada teve um impacto **moderado** na redução do erro percentual simétrico, o que é benéfico, pois evita que o modelo favoreça previsões tendenciosas em relação a valores altos ou baixos.

**Interpretação**: A função de perda testada **reduziu o erro simétrico** de forma significativa, mas a melhoria foi mais modesta quando comparada ao MAPE.

---

### **Conclusão Geral**
- **Métrica com maior melhoria**: **MAPE (59.44%)**, o que sugere que a função de perda testada teve um grande impacto na precisão percentual das previsões.
- **Métrica com menor melhoria**: **MAE (2.89%)**, que teve uma melhoria modesta, indicando que o erro absoluto médio não foi reduzido de forma tão significativa quanto as outras métricas.
- **Impacto das funções de perda**: O modelo demonstrou uma melhoria significativa no desempenho com a função de perda testada, especialmente nas métricas de erro percentual (**MAPE e SMAPE**), o que pode ser crucial dependendo do tipo de aplicação (como previsão de demanda ou vendas).

**Conclusão final**: A função de perda testada mostrou-se eficaz em melhorar as previsões, especialmente em termos relativos (MAPE e SMAPE), o que pode ser desejável para muitos cenários. Se o foco for **minimizar os grandes erros** ou melhorar a previsão percentual, a função de perda escolhida parece ser a **melhor opção** entre as testadas.
''')
    
st.header('Forecast TimeGPT para Licenciamentos Automóveis Aprimorado', divider='green')
col1, col2 = st.columns([1,1], gap='large')
with col1:
    st.write('Nova Previsão FineTuning_Steps = 60; FineTuning_Mae e FineTuning_depths = 1')
    st.dataframe(timegpt_fcst_finetune_mae_df, height=500)
    st.markdown('''O código fornecido está utilizando a biblioteca **Nixtla** (presumivelmente para previsão de séries temporais) para realizar previsões utilizando o modelo **TimeGPT** com ajuste fino (**fine-tuning**) de parâmetros, com o foco na **função de perda MAE**. Vamos analisar cada parte do código e explicar sua função:

---

### **1. Definição da Função `forecast`**

A função `nixtla_client.forecast()` é usada para gerar previsões de séries temporais, utilizando diferentes parâmetros para ajustar o modelo conforme a necessidade. O retorno dessa função é um **DataFrame** com as previsões feitas pelo modelo.

---

### **2. Parâmetros do Modelo de Previsão**

- **`df=df_long`**: 
  - O parâmetro `df` representa o conjunto de dados que será utilizado para treinar e gerar previsões. No caso, `df_long` é um DataFrame com os dados da série temporal, onde `ds` representa a coluna de data e `y` a variável alvo (o valor que se deseja prever).

- **`h=36`**:
  - O parâmetro `h` define o **horizonte de previsão**, ou seja, o número de períodos para os quais o modelo irá fazer as previsões. No caso, o modelo está configurado para prever os próximos **36 períodos** (o que pode representar, por exemplo, 36 meses ou 36 dias, dependendo da granularidade dos dados).

- **`finetune_steps=60`**:
  - Esse parâmetro especifica o número de **passos de ajuste fino (fine-tuning)** que o modelo realizará para melhorar a precisão das previsões. O modelo será ajustado durante **60 iterações**, o que permite que o modelo se especialize mais nas particularidades dos dados.

- **`finetune_loss='mae'`**:
  - A função de perda a ser usada durante o processo de ajuste fino. O parâmetro `'mae'` significa que o modelo será otimizado para minimizar o **Erro Absoluto Médio (MAE - Mean Absolute Error)**. O MAE mede a média dos erros absolutos entre as previsões e os valores reais, ou seja, é uma métrica simples e interpretável, adequada para medir a precisão do modelo em termos absolutos.

- **`finetune_depth=1`**:
  - A profundidade do ajuste fino. Esse parâmetro pode controlar a complexidade do modelo durante o treinamento adicional. Um valor de **1** geralmente significa um ajuste mais leve e rápido, enquanto valores maiores indicam um ajuste mais profundo e complexo.

- **`time_col='ds'`** e **`target_col='y'`**:
  - Esses parâmetros especificam as colunas do DataFrame que representam, respectivamente, o **tempo (ds)** e a variável alvo (**y**). O modelo usa essas colunas para entender as relações entre os dados históricos e fazer previsões.

- **`level=[80, 90, 95]`**:
  - Esse parâmetro define os **níveis de confiança** para as previsões. No caso, as previsões serão feitas com intervalos de confiança de **80%**, **90%** e **95%**. Isso permite que você tenha uma ideia da **variabilidade das previsões**, fornecendo uma faixa de possíveis valores para o futuro, em vez de uma previsão única.

- **`add_history=True`**:
  - O parâmetro `add_history=True` indica que o modelo deve **incluir dados históricos adicionais** ao realizar as previsões. Isso é útil para capturar padrões sazonais e tendências que podem estar presentes no passado.

---

### **3. Objetivo do Código**

O objetivo desse código é treinar o modelo **TimeGPT** utilizando dados históricos e realizar previsões para os próximos 36 períodos. Durante o treinamento, o modelo passará por **60 iterações de ajuste fino**, sendo otimizado com a função de perda **MAE**. O modelo também gera intervalos de confiança para suas previsões, oferecendo uma estimativa mais robusta para os próximos valores.

---

### **Vantagens dessa Abordagem**

1. **Ajuste Fino (Fine-tuning)**: 
   - O modelo é ajustado para otimizar uma métrica de erro específica (neste caso, o MAE), o que pode melhorar significativamente a qualidade das previsões em relação ao modelo básico.

2. **Previsões com Intervalos de Confiança**:
   - Ao fornecer previsões com diferentes níveis de confiança (**80%, 90%, 95%**), o modelo permite uma análise mais abrangente, considerando a incerteza nas previsões.

3. **Consideração de Dados Históricos**:
   - Ao incluir os dados históricos no processo de previsão (`add_history=True`), o modelo pode capturar melhor as tendências e padrões sazonais, o que pode melhorar a precisão das previsões.

4. **Otimização para MAE**:
   - O MAE é uma função de perda intuitiva e muito utilizada em problemas de previsão, pois mede diretamente a média dos erros absolutos, sem penalizar excessivamente os grandes erros (ao contrário do MSE). Isso pode ser útil quando os erros de grande magnitude são indesejáveis.

---

### **Conclusão**

Esse código utiliza o **TimeGPT** para gerar previsões ajustadas com base em dados históricos, com o processo de ajuste fino otimizando o modelo de acordo com o **Erro Absoluto Médio (MAE)**. As previsões geradas vêm com **intervalos de confiança**, oferecendo uma visão mais completa e robusta sobre o futuro, ao mesmo tempo que são otimizadas para minimizar o erro absoluto. Essa abordagem é eficaz para melhorar a precisão do modelo e fornecer previsões com maior confiabilidade.
''')

with col2:
    st.write('Gráfico Forecast TimeGPT Aprimorados para Licenciamentos Automóveis')
    st.image('C:\\Tablets\\fcstfine.png', width=1000)
    st.image('C:\\Tablets\\codigo.png', width= 1000)




