import streamlit as st
import pandas as pd
import pickle
import json

# Функция загрузки модели
def load_model():
    with open('/app/data/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title('Предсказание вида Ирисов')
st.write('Пожалуйста, загрузите JSON-файл, содержащий характеристики Ирисов для предсказания.')

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выберите JSON файл", type=['json'])
if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
        df = pd.DataFrame([data])

        # Проверка наличия необходимых колонок
        required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        if all(column in df.columns for column in required_columns):
            # Предсказания
            prediction = model.predict(df[required_columns])
            st.write('Предсказанный вид Ирисов:', prediction[0])
        else:
            st.error(f"JSON-файл должен содержать следующие колонки: {required_columns}")

    except json.JSONDecodeError:
        st.error("Ошибка чтения JSON файла. Пожалуйста, убедитесь, что файл имеет правильный формат JSON.")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
    finally:
        uploaded_file.seek(0)   # Возвращаемся к началу файла, если потребуется повторное чтение