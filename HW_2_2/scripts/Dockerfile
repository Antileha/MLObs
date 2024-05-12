# Используем официальный образ Jenkins#-jdk17
FROM jenkins/jenkins:lst 

# Переключаемся на пользователя root для учтановки пакетов
USER root

# Обновление списка пакетов и установка Python и pip
RUN apt-get update && \
    apt_get install -y python3 python3-pip python3-venv 

# Создание виртуального окружения 
RUN python3 -m venv /opt/venv


# Активация виртуального окружения и установка пакетов
ENV PATH= "/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install numpy pandas scikit-learn matplotlib seaborn jupyter 


# Возвращаем пользователя Jenkins
USER jenkins
