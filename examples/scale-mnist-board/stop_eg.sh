ps -ef |grep client.py |grep -v grep |awk '{print "kill -9 "$2}' | sh
ps -ef |grep server.py |grep -v grep |awk '{print "kill -9 "$2}' | sh
ps -ef |grep board.py |grep -v grep |awk '{print "kill -9 "$2}' | sh