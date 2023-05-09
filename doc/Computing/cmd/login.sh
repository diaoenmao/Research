screen -dm -S upload
screen -dm -S download
for i in {01..14}
do
	screen -dm -S duke$i bash -c "ssh -i ./ssh/id_rsa ed155@research-tarokhlab-$i.oit.duke.edu; exec sh"
done