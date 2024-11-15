set -x

function pull_scm_binary() {
    local scm_psm=$1
    local scm_path=$2
    local scm_version=$3 # version/""
    local scm_action=$4  # ""/force/purge

    echo "::add-message::installing $scm_psm to $scm_path ..."

    [[ ! "$scm_path" =~ ^/opt/tiger/ ]] && echo "::add-message::only support pulling scm binary under /opt/tiger/" && return
    [[ "$scm_action" == "purge" ]] && (rm -rf $scm_path || sudo rm -rf $scm_path)
    [[ "$scm_action" != "force" && -e $scm_path ]] && { echo "::add-message::$scm_psm already exists, $(cat $scm_path/current_revision | grep "version")"; return; }

    # ensure tiger permission to pull binary
    [[ ! -e $scm_path ]] && (mkdir -p $scm_path || sudo mkdir -p $scm_path)
    chown -R tiger:tiger $scm_path || sudo chown -R tiger:tiger $scm_path

    [[ -n "$scm_version" ]] && scm_version="--version $scm_version"
    sudo -u tiger -g tiger XDG_RUNTIME_DIR=/run/user/1000 /opt/tiger/bvc/bin/bvc clone $scm_psm $scm_path $scm_version -f && echo "::add-message::$scm_psm installed, $(cat $scm_path/current_revision | grep "version")"
}

cd /opt/tiger/
ls -alh

pull_scm_binary "jdk" "/opt/tiger/jdk" "" ""

export JAVA_HOME=/opt/tiger/jdk/jdk1.8
echo "::set-env name=JAVA_HOME::$JAVA_HOME"

pull_scm_binary "yarn_deploy" "/opt/tiger/yarn_deploy" "" "force"

echo "::add-message::installing hadoop ..."
su - tiger -c "rm -rf /opt/tiger/yarn_deploy/hadoop/conf && /opt/tiger/yarn_deploy/hadoop/bin/hadoop"
ls /opt/tiger/yarn_deploy/hadoop/bin/

pull_scm_binary "data/inf/hdfs_client" "/opt/tiger/hdfs_client" "" "purge"
pull_scm_binary "lab/arnold/hdfs_client" "/opt/tiger/arnold/hdfs_client" "" "purge"
