export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/jdk/jdk1.8/jre/lib/amd64/server
export HADOOP_HOME=/opt/tiger/yarn_deploy/hadoop
export CLASSPATH=`${HADOOP_HOME}/bin/hadoop classpath --glob`
export PATH=/opt/tiger/yarn_deploy/hadoop/bin/:$PATH
export RUNTIME_IDC_NAME=$(cat /opt/tmp/consul_agent/datacenter)
