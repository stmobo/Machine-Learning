#!/bin/bash
RED='\033[1;31m'
GREEN='\033[1;32m'
GREY='\033[0;37m'
YELLOW='\033[1;33m'

get-pid() {
    if [[ $1 == `uname -n` ]]; then
        PID=$(pgrep -f 'python3 /mnt/fnst/fast-nst/fast-style-transfer.py');
    else
        PID=$(ssh stmobo@$1 pgrep -f \'python3 /mnt/fnst/fast-nst/fast-style-transfer.py\');
    fi
}

check-nodes() {
    for node in "$@"; do
    	echo -n "$node:";
    	get-pid "$node";

    	if [[ -n ${PID} ]]; then
    		echo -e "${GREEN} Running${GREY} (PID ${PID}).";
    	else
    		echo -e "${RED} Not running${GREY}.";
    	fi
    done
}

stop-nodes() {
    for node in "$@"; do
        echo -n "$node:";
        get-pid "$node";

        if [[ -z ${PID} ]]; then
            echo -e " ${YELLOW}Not running${GREY}.";
        else
            if [[ $node == `uname -n` ]]; then
                kill ${PID}
            else
	            ssh stmobo@$node kill ${PID}
            fi

            if [[ $? -eq 0 ]]; then
                echo -e "${GREEN} Success${GREY} (killed ${PID}).";
            else
                echo -e "${RED} Failed${GREY} (PID: ${PID}).";
            fi
        fi
    done
}

start-nodes() {
    for node in "$@"; do
        echo -n "$node:";
    	get-pid "$node";
    	if [[ -z ${PID} ]]; then
            if [[ $node == `uname -n` ]]; then
                /bin/bash /mnt/fnst/run-scripts/run-node.sh;
                sleep 1;
            else
	            ssh stmobo@$node /bin/bash /mnt/fnst/run-scripts/run-node.sh;
            fi

    		if [[ $? -eq 0 ]]; then
    			get-pid "$node";
    			echo -e "${GREEN} Success${GREY} (PID ${PID}).";
    		else
    			echo -e "${RED} Failed${GREY}. (status $?)";
    		fi
    	else
    		echo -e "${YELLOW} Already running${GREY} (PID ${PID}).";
    	fi
    done
}

# Action in $1, target(s) in the rest of the parameters
nodes=('fnst-ps-0' 'fnst-ps-1' 'fnst-worker-1' 'fnst-worker-2' 'fnst-worker-3' 'fnst-worker-4' 'fnst-chief')
if [[ $2 == 'ps' ]]; then # parameter servers
    nodes=('fnst-ps-0' 'fnst-ps-1')
elif [[ $2 == 'workers' ]]; then
    nodes=('fnst-worker-1' 'fnst-worker-2' 'fnst-worker-3' 'fnst-worker-4' 'fnst-chief')
elif [[ -n $2 ]]; then
    nodes="$@";
fi

if [[ $1 == 'start' ]]; then
    start-nodes "${nodes[@]}";
elif [[ $1 == 'stop' ]] || [[ $1 == 'kill' ]]; then
    stop-nodes "${nodes[@]}";
elif [[ $1 == 'restart' ]]; then
    stop-nodes "${nodes[@]}";
    start-nodes "${nodes[@]}";
elif [[ $1 == 'check' ]] || [[ -z $1 ]]; then
    check-nodes "${nodes[@]}";
else
    echo "nodes.sh: nrecognized command $1 (valid options are 'start', 'stop', 'check')";
fi
