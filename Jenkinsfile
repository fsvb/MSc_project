pipeline {
  agent any

  parameters {
    string(name: 'DOCKER_IMG', defaultValue: 'tensorflow/tensorflow:1.10.1-gpu-py3', description: 'Docker image with TensorFlow')
    string(name: 'CMD_LINE', defaultValue: '--model resnet_18 --dataset cifar10', description: 'Command line parameters to runner script')

    // choice(choices: 'yes\nno\n',
    //        description: 'Destroy TensorBoard after training', name: 'PARAM_DESTROY_TENSORBOARD')
  }

  stages {
    stage('Pre-Training') {
      steps {
        sh "hostname"
        sh "printenv"
        sh "pwd"
      }
    }

    stage('Training') {
      agent {
        docker {
            image '$DOCKER_IMG'
            args '''--runtime=nvidia \
                    -e NVIDIA_VISIBLE_DEVICES=$EXECUTOR_NUMBER \
                    --name=$BUILD_TAG'''
            reuseNode true /* Without this the contianer can't see the workspace. Works across multiple machines too */
            alwaysPull true
        }
      }

      steps {
        parallel(
          "Main training loop": {
            sh "python -u main.py $CMD_LINE"
          }
//          "Query nvidia-smi": {
//            sh "bash -c 'while true; do nvidia-smi; sleep 20; done'"
//          }
        )
      }
    }

    // stage('Stop TensorBoard') {
    //   when {
    //     environment name: 'PARAM_DESTROY_TENSORBOARD', value: 'yes'
    //   }
    //   steps {
    //     sh 'docker rm --force tensorboard_"$BUILD_TAG"'
    //   }
    // }

    stage('Post-Training') {
      steps {
        sh "ls -l"
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: '*.csv', fingerprint: true
      deleteDir() /* clean up our workspace */
    }

    success {
      slackSend channel: '#jenkins', color: 'good',
        message: "The pipeline *${currentBuild.fullDisplayName}* (Build #${currentBuild.number}) completed successfully after ${currentBuild.durationString} <$RUN_DISPLAY_URL|Open>"
    }

    failure {
      slackSend channel: '#jenkins', color: 'danger',
        message: "The pipeline *${currentBuild.fullDisplayName}* (Build #${currentBuild.number}) failed after ${currentBuild.durationString}!! <$RUN_DISPLAY_URL|Open>"
    }
  }
}
