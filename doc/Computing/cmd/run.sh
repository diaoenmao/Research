ssh -i ./ssh/id_rsa ed155@research-tarokhlab-10.oit.duke.edu "bash train_joint_ML100K_1.sh; bash train_alone_ML100K_1.sh; bash train_assist_ML100K_1.sh"
ssh -i ./ssh/id_rsa ed155@research-tarokhlab-11.oit.duke.edu "bash train_joint_ML1M_1.sh; bash train_alone_ML1M_1.sh; bash train_assist_ML1M_1.sh"
ssh -i ./ssh/id_rsa ed155@research-tarokhlab-12.oit.duke.edu "bash train_joint_ML10M_1.sh; bash train_alone_ML10M_1.sh; bash train_assist_ML10M_1.sh"
ssh -i ./ssh/id_rsa ed155@research-tarokhlab-13.oit.duke.edu "bash train_joint_Douban_1.sh; bash train_alone_Douban_1.sh; bash train_assist_Douban_1.sh"
ssh -i ./ssh/id_rsa ed155@research-tarokhlab-14.oit.duke.edu "bash train_joint_Amazon_1.sh; bash train_alone_Amazon_1.sh; bash train_assist_Amazon_1.sh"


# ssh -i ./ssh/id_rsa ed155@research-tarokhlab-10.oit.duke.edu "bash test_joint_ML100K_1.sh; bash test_alone_ML100K_1.sh; bash test_assist_ML100K_1.sh"
# ssh -i ./ssh/id_rsa ed155@research-tarokhlab-11.oit.duke.edu "bash test_joint_ML1M_1.sh; bash test_alone_ML1M_1.sh; bash test_assist_ML1M_1.sh"
# ssh -i ./ssh/id_rsa ed155@research-tarokhlab-12.oit.duke.edu "bash test_joint_ML10M_1.sh; bash test_alone_ML10M_1.sh; bash test_assist_ML10M_1.sh"
# ssh -i ./ssh/id_rsa ed155@research-tarokhlab-13.oit.duke.edu "bash test_joint_Douban_1.sh; bash test_alone_Douban_1.sh; bash test_assist_Douban_1.sh"
# ssh -i ./ssh/id_rsa ed155@research-tarokhlab-14.oit.duke.edu "bash test_joint_Amazon_1.sh; bash test_alone_Amazon_1.sh; bash test_assist_Amazon_1.sh"
