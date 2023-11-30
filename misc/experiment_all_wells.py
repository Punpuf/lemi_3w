#!/usr/bin/env python
# coding: utf-8

# # Experimento com dados de todos os poços
# 
# Nesse notebook será realizado as tarefas de obtenção de dados, tratamento, modelagem, e validação de dados.
# O objetivo aqui é obter um classificador de anomalias para dados de todos os poços (aqui será incluido todos os tipos de dados: inclusive dados de fontes simulação e desenhados).

# # Aquisição de dados

# Configurando ambiente: 

# In[1]:


import sys
sys.path.append(".")

# Environment configuration
import raw_data_manager.raw_data_acquisition as rda
import raw_data_manager.raw_data_inspector as rdi
import raw_data_manager.raw_data_splitter as rds
from data_exploration.metric_acquisition import MetricAcquisition
from data_preparation.transformation_manager import TransformationManager
from constants import storage_config
import pathlib
import numpy as np
import random
import pickle

# Set default logging level.
from absl import logging
logging.set_verbosity(logging.DEBUG)


# Baixar dataset 3W (se não disponível) & gerar tabela de metadados.

# In[2]:


## Acquire data (of entire 3W dataset)
rda.acquire_dataset_if_needed()
latest_converted_data_path, latest_converted_data_version = (
    rda.get_latest_local_converted_data_version(storage_config.DIR_PROJECT_DATA)
)

# Helper to overview metadata (of entire 3W dataset)
inspector_all_data = rdi.RawDataInspector(
    dataset_dir=latest_converted_data_path,
    cache_file_path=storage_config.PATH_DATA_INSPECTOR_CACHE,
    use_cached=True
)
metadata_all_data = inspector_all_data.get_metadata_table()
metadata_all_data


# Dividir dados (de forma estratificada) em treinamento e teste.

# In[3]:


# splits data, from the selected well, into train and test datasets
splitter = rds.RawDataSplitter(metadata_all_data, latest_converted_data_version)
split_train_dir, split_validation_dir, split_test_dir = splitter.stratefy_split_of_data(
    data_dir=storage_config.DIR_PROJECT_DATA, 
    validation_size=0.10,
    test_size=0.20,
)


# In[4]:


# generates metadata tables for split data
train_metadata = rdi.RawDataInspector(
    dataset_dir=split_train_dir,
    cache_file_path=storage_config.DIR_PROJECT_CACHE / "train_metadata_all_data.parquet",
    use_cached=False,
)
validation_metadata = rdi.RawDataInspector(
    dataset_dir=split_validation_dir,
    cache_file_path=storage_config.DIR_PROJECT_CACHE / "validation_metadata_all_data.parquet",
    use_cached=False,
)
test_metadata = rdi.RawDataInspector(
    dataset_dir=split_test_dir,
    cache_file_path=storage_config.DIR_PROJECT_CACHE / "test_metadata_all_data.parquet",
    use_cached=False,
)


# Tabela de anomalias por tipo de fonte - treinamento.

# In[5]:


rdi.RawDataInspector.generate_table_by_anomaly_source(train_metadata.get_metadata_table())


# Tabela de anomalias por tipo de fonte - validação.

# In[6]:


rdi.RawDataInspector.generate_table_by_anomaly_source(validation_metadata.get_metadata_table())


# Tabela de anomalias por tipo de fonte - teste.

# In[7]:


rdi.RawDataInspector.generate_table_by_anomaly_source(test_metadata.get_metadata_table())


# # Procesamento de dados

# Remoção de valores extremos.

# In[8]:


from parallelbar import progress_map

def is_event_path_values_valid(event_path):
    event = rda.get_event(event_path)
    return TransformationManager.is_event_values_valid(event)

valid_index_list = progress_map(is_event_path_values_valid, train_metadata.get_metadata_table()['path'].to_list())

non_valid_ids = [index for index, value in enumerate(valid_index_list) if value == False]
non_valid_ids


# In[9]:


valid_train_metadata = train_metadata.get_metadata_table().copy()
print("stating size is:", len(valid_train_metadata))

valid_train_metadata.drop(valid_train_metadata.index[non_valid_ids], inplace=True, axis='index')
print("final size is:", len(valid_train_metadata))


# Calcular valores da média dos valores e do desvio padrão.

# In[10]:


from data_exploration.metric_acquisition import MetricAcquisition

cache_file_name = "all_data_metrics"

metric_aquisition = MetricAcquisition(valid_train_metadata)
mean_and_std_metric_table = metric_aquisition.get_mean_and_std_metric(
    cache_file_name=cache_file_name,
    use_cache=False
)

mean_metric_list = mean_and_std_metric_table['mean_of_means']
std_metric_list = mean_and_std_metric_table['mean_of_stds']
mean_and_std_metric_table


# Realizar a transformação dos dados.

# In[11]:


selected_variables = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", 
                      "T-JUS-CKP", "P-JUS-CKGL", "class"]
transformation_param_sample_interval_seconds=60
transformation_param_num_timesteps_for_window=20


# Transformar conjunto de dados para treinamento.

# In[12]:


train_tranformed_folder_name = split_train_dir.name

train_transformation_manager = TransformationManager(
    valid_train_metadata, 
    output_folder_base_name=train_tranformed_folder_name
)

train_transformation_manager.apply_transformations_to_table(
    output_parent_dir=storage_config.DIR_PROJECT_DATA,
    selected_variables=selected_variables,
    sample_interval_seconds=transformation_param_sample_interval_seconds,
    num_timesteps_for_window=transformation_param_num_timesteps_for_window,
    avg_variable_mean=mean_metric_list,
    avg_variable_std_dev=std_metric_list,
)


# Transformar conjunto de dados para validação.

# In[13]:


validation_tranformed_folder_name = split_validation_dir.name

validation_transformation_manager = TransformationManager(
    validation_metadata.get_metadata_table(), 
    output_folder_base_name=validation_tranformed_folder_name
)

validation_transformation_manager.apply_transformations_to_table(
    output_parent_dir=storage_config.DIR_PROJECT_DATA,
    selected_variables=selected_variables,
    sample_interval_seconds=transformation_param_sample_interval_seconds,
    num_timesteps_for_window=transformation_param_num_timesteps_for_window,
    avg_variable_mean=mean_metric_list,
    avg_variable_std_dev=std_metric_list,
)


# Transformar conjunto de dados para testagem.

# In[14]:


test_tranformed_folder_name = split_test_dir.name

test_transformation_manager = TransformationManager(
    test_metadata.get_metadata_table(),
    output_folder_base_name=test_tranformed_folder_name
)

test_transformation_manager.apply_transformations_to_table(
    output_parent_dir=storage_config.DIR_PROJECT_DATA,
    selected_variables=selected_variables,
    sample_interval_seconds=transformation_param_sample_interval_seconds,
    num_timesteps_for_window=transformation_param_num_timesteps_for_window,
    avg_variable_mean=mean_metric_list,
    avg_variable_std_dev=std_metric_list,
)


# # Verificação dos dados
# 
# Garantir que os dados processados estão dentro das especificações esperadas.

# Obter lista dos arquivos a serem usados no treinamento, validação e testagem.

# In[15]:


# Get transformed files paths
TRANSFORMATION_NAME_PREFIX = "transform-isdt-"
train_tranformed_dataset_dir = storage_config.DIR_PROJECT_DATA / (TRANSFORMATION_NAME_PREFIX + train_tranformed_folder_name)
validation_tranformed_dataset_dir = storage_config.DIR_PROJECT_DATA / (TRANSFORMATION_NAME_PREFIX + validation_tranformed_folder_name)
test_tranformed_dataset_dir = storage_config.DIR_PROJECT_DATA / (TRANSFORMATION_NAME_PREFIX + test_tranformed_folder_name)

# Generate inspectors for the transformed data
train_inspector_converted = rdi.RawDataInspector(
    train_tranformed_dataset_dir,
    storage_config.DIR_PROJECT_CACHE / "all_wells_transformed-train.parquet",
    use_cached=False
)
validation_inspector_transformed = rdi.RawDataInspector(
    validation_tranformed_dataset_dir,
    storage_config.DIR_PROJECT_CACHE / "all_wells_transformed-validation.parquet",
    use_cached=False
)
test_inspector_transformed = rdi.RawDataInspector(
    test_tranformed_dataset_dir,
    storage_config.DIR_PROJECT_CACHE / "all_wells_transformed-test.parquet",
    use_cached=False
)

# Get list of paths for the events in each of the data groups (train, val, test)
train_metadata_converted = train_inspector_converted.get_metadata_table()
train_transformed_file_path_list = train_metadata_converted["path"].to_list()

validation_metadata_transformed = validation_inspector_transformed.get_metadata_table()
validation_transformed_file_path_list = validation_metadata_transformed["path"].to_list()

test_metadata_transformed = test_inspector_transformed.get_metadata_table()
test_transformed_file_path_list = test_metadata_transformed["path"].to_list()

# shuffle
random.shuffle(train_transformed_file_path_list)
random.shuffle(validation_transformed_file_path_list)
random.shuffle(test_transformed_file_path_list)


# Exemplo de um par X, y.

# In[16]:


example_transformed_file_path = train_transformed_file_path_list[0]
X_transformed_ex, y_transformed_ex = TransformationManager.retrieve_pair_array(pathlib.Path(example_transformed_file_path))

X_transformed_ex[0], y_transformed_ex[0]


# Verificar valores mínimos e máximos dos dados.

# In[17]:


X_min_list = []
X_max_list = []
y_avg = []

for path in train_transformed_file_path_list:
    X, y = TransformationManager.retrieve_pair_array(pathlib.Path(path))
    X_min_list.append(np.min(np.min(X, axis=0), axis=0))
    X_max_list.append(np.max(np.max(X, axis=0), axis=0))
    y_avg.append(np.sum(y, axis=0) / len(y))

X_min_list = np.array(X_min_list)
X_max_list = np.array(X_max_list)
y_avg = np.array(y_avg)


# In[18]:


# Regarding minimum values of X
print("Minimum minimum values:\n", X_min_list.min(axis=0))

print("\n\nMinimum average values:\n", X_min_list.mean(axis=0))

print("\n\nMinimum maximum values:\n", X_min_list.max(axis=0))


# In[19]:


# Regarding max values of X
print("Maximum minimum values:\n", X_max_list.min(axis=0))

print("\n\nMaximum average values:\n", X_max_list.mean(axis=0))

print("\n\nMaximum maximum values:\n", X_max_list.max(axis=0))


# In[20]:


import matplotlib.pyplot as plt

variable_index = 0
plt.scatter(X_max_list[:, variable_index], X_max_list[:, variable_index], alpha=0.2)


# In[21]:


# Regarding mean values of y
print("Mean y minimum values:\n", y_avg.min(axis=0))

print("\n\nMean y average values:\n", y_avg.mean(axis=0))

print("\n\nMean y maximum values:\n", y_avg.max(axis=0))


# In[22]:


y_avg.sum(axis=0)


# # Modelagem

# In[23]:


from tensorflow import keras, math
import keras_tuner
from raw_data_manager.models import EventClassType


# Função responsável por gerar modelos de acordo com uma lista de hyperparâmetros de entrada.

# In[24]:


num_features = X_transformed_ex.shape[2]
num_outputs = len(EventClassType)

def build_model(hp: keras_tuner.HyperParameters):
    model_type = hp.Choice("model_type", 
                           ["lstm-vanilla", 
                            "lstm-stacked", 
                            "lstm-bidirectional"])

    model = keras.Sequential()
    match model_type:
        case "lstm-vanilla":
            model.add(keras.layers.LSTM(
                hp.Int('lstm_vanilla_units', min_value=5, max_value=100, step=5), 
                activation='relu', 
                input_shape=(transformation_param_num_timesteps_for_window, num_features)))
        case "lstm-stacked":
            model.add(keras.layers.LSTM(
                hp.Int('lstm_stacked_1_units', min_value=5, max_value=100, step=5), 
                activation='relu', 
                return_sequences=True,
                input_shape=(transformation_param_num_timesteps_for_window, num_features)))
            
            model.add(keras.layers.LSTM(
                hp.Int('lstm_stacked_2_units', min_value=5, max_value=100, step=5), 
                activation='relu', ))
        case "lstm-bidirectional":
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                hp.Int('lstm_bidirectional_units', min_value=5, max_value=100, step=5), 
                activation='relu', 
                input_shape=(transformation_param_num_timesteps_for_window, num_features))))
            
    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Dense(
        hp.Int('dense_units', min_value=20, max_value=200, step=10),
        activation='relu'))
    
    model.add(keras.layers.Dense(num_outputs, activation='softmax'))

    # compiling model
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-1, sampling="log")
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        metrics=['accuracy'],
    )
    return model

# check the model can compile
build_model(keras_tuner.HyperParameters())


# In[25]:


tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_accuracy",
    max_epochs=9, # experiments for each model
    hyperband_iterations=3, # times to cycle through the hyperband algorithm
    seed=1331,
    overwrite=True,
    directory=storage_config.DIR_PROJECT_DATA / "keras_tuner",
    project_name="lstm_vanilla",
)

tuner.search_space_summary()


# Realizar a pesquisa pelos melhores hyperparâmetros.

# In[26]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')
#get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[27]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')

#get_ipython().run_line_magic('tensorboard', '--logdir /tmp/tb_logs')


# In[30]:


import datetime
current_run_datatime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = pathlib.Path("logs") / current_run_datatime
logdir.mkdir(parents=True, exist_ok=True)

num_epochs = 5
steps_per_epoch = len(train_transformed_file_path_list)

train_data_gen = TransformationManager.data_generator_loop(train_transformed_file_path_list)
validation_data_gen = TransformationManager.data_generator_loop(validation_transformed_file_path_list)
test_data_gen = TransformationManager.data_generator_loop(test_transformed_file_path_list)

tuner.search(
    train_data_gen,
    validation_data=validation_data_gen,
    validation_steps=len(validation_transformed_file_path_list),
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    callbacks=[keras.callbacks.TensorBoard(logdir)],
)


# In[34]:


# Safe to cache best found parameters
path_best_params = storage_config.DIR_PROJECT_CACHE / "param_search" / f"all_data_{current_run_datatime}.pkl"
path_best_params.parent.mkdir(parents=True, exist_ok=True)
best_params = tuner.get_best_hyperparameters(5)[0]

with open(path_best_params,'wb') as file:
    pickle.dump(best_params, file)

best_params.values


# In[35]:


best_params.values


# Utilizar os parâmetros obtidos que tiveram os melhores resultados para treinar um modelo.
# 
# Dessa vez, o modelo será treinado com uma quantidade maior de épocas.

# In[37]:


# Training configurations
num_epochs = 50
steps_per_epoch = len(train_transformed_file_path_list)
train_data_gen = TransformationManager.data_generator_loop(train_transformed_file_path_list)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

checkpoint_filepath = storage_config.DIR_PROJECT_CACHE / "model_weights" / "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train model
best_model = build_model(best_params)

training_history = best_model.fit(
    train_data_gen, 
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=validation_data_gen,
    validation_steps=len(validation_transformed_file_path_list),
    callbacks=[early_stopping_callback, model_checkpoint_callback],
    verbose=1
)

# Graph of training results history
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Ver se está funcionando com um exemplo.

# In[ ]:


import numpy as np
test_file = train_transformed_file_path_list[-1]
print(test_file)

Xhat, yhat = TransformationManager.retrieve_pair_array(pathlib.Path(test_file))
print(f"True value: {yhat[0]}")

Xhat0 = Xhat[0].reshape(1, transformation_param_num_timesteps_for_window, num_features)
print(f"Predicted value: {best_model.predict(Xhat0)}")


# # Validação
# Aqui pegaremos nosso banco de testes, o transformaremos, para então o utilizar para validar a perfomance do nosso modelo.

# In[ ]:


test_data_gen = TransformationManager.data_generator_non_loop(test_transformed_file_path_list)
num_steps = len(test_transformed_file_path_list)

best_model.evaluate(
    test_data_gen,
    steps=num_steps,
    verbose=1,
)


# In[ ]:


test_data_gen = TransformationManager.data_generator_non_loop(test_transformed_file_path_list)

y_test_predictions = best_model.predict(
    test_data_gen,
)

(
    f"Number of predictions: {len(y_test_predictions)}", 
    f"Shape of y array: {y_test_predictions.shape}", 
    y_test_predictions[0]
)


# In[ ]:


test_data_gen = TransformationManager.data_generator_non_loop(test_transformed_file_path_list)
y_test_labels = []

for X, y in test_data_gen:
    y_test_labels.append(y)

y_test_labels = np.concatenate(y_test_labels, axis=0)

(
    f"Number of predictions: {len(y_test_labels)}", 
    f"Shape of y array: {y_test_labels.shape}", 
    y_test_labels[0],
)


# In[ ]:


y_test_labels_1d = np.argmax(y_test_labels, axis=1)
y_test_predictions_1d = np.argmax(y_test_predictions, axis=1)

math.confusion_matrix(
    y_test_labels_1d,
    y_test_predictions_1d,
    num_classes=num_outputs,
)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from raw_data_manager.models import EventClassType

consusion_array = np.array([[31146,  2060,    78,     7,   603,   457,   473,   883,   233],
       [  669, 25849,     0,  1091,     0,     0,     0,  2973,     0],
       [   31,    20,  1379,    25,     0,     2,     3,     0,    14],
       [    0,     0,     0, 13869,   278,  1009,     0,     0,    15],
       [  975,     0,     0,     1,  5981,     0,     0,     0,     0],
       [    3,    13,     0,    87,    20, 41808,     0,     0,     0],
       [   11,     0,     0,     0,     0,     0, 18060,     0,     0],
       [    0,     0,     0,     0,     0,     0,     0,  5790,     0],
       [  111,     0,     0,     0,    15,  1102,     0,     0,  5492]],)

fig, ax = plt.subplots(figsize=(8,6))
heatmap = sns.heatmap(consusion_array, annot=True, fmt='d', xticklabels=[e.value for e in EventClassType], yticklabels=[e.value for e in EventClassType], annot_kws={"size": 12})
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 14)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 14)


plt.ylabel('Real values', fontsize=18)
plt.xlabel('Predicted values', fontsize=19)
plt.show()

