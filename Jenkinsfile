pipeline {
    agent any

    stages {
        stage('Setup Environment') {
            steps {
                sh 'echo "Setting up environment..."'
                // Установка зависимостей, если это необходимо
            }
        }
        stage('Download Data') {
            steps {
                sh 'python3 HW_2_2/scripts/download_data.py'
            }
        }
        stage('Preprocess Data') {
            steps {
                sh 'python3 HW_2_2/scripts/preprocess.py'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python3 HW_2_2/scripts/train_model.py'
            }
        }
        stage('Test Model') {
            steps {
                sh 'python3 HW_2_2/scripts/test_model.py'
            }
        }
    }

    post {
        always {
            sh 'echo "Pipeline completed."'
        }
    }
}
