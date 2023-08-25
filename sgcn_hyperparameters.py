isShareAdj = False
isInfo_Score = True
isSeperatedGender = False
selected_gender = 1

epochs = 200
N = 90
H_0 = 3
H_1 = 5
H_2 = 5
H_3 = 5
train_frac = 0.8
batch_size = 32
shuffle_seed = 1000
learning_rate = 0.0005
droupout_prob = 0.5

lamda_x_l1=0.1
lamda_e_l1=0.1
lamda_x_ent=0.1
lamda_e_ent=0.1
lamda_mi=1
lamda_ce=1


def string():
    str_result = ('epochs:%d\nH_0:%d\nH_1:%d\nH_2:%d\nH_3:%d\nbatch_size:%d\nshuffle_seed:%d\nlearning_rate:%f\nlamda_x_l1:%f\nlamda_e_l1:%f\nlamda_x_ent:%f\nlamda_e_ent:%f\nlamda_mi:%f\nlamda_ce:%f\ndroupout_prob:%f\n'\
          %(epochs,H_0,H_1,H_2,H_3,batch_size,shuffle_seed,learning_rate,lamda_x_l1,lamda_e_l1,lamda_x_ent,lamda_e_ent,lamda_mi,lamda_ce,droupout_prob))
    print(str_result)
    return str_result