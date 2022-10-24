python large_scale_data_preparation.py #to prepare data

bash large_scale_train_server_1.sh; bash large_scale_test_server_1.sh #on the 1th server

bash large_scale_train_server_2.sh; bash large_scale_test_server_2.sh #on the 2th server

bash large_scale_train_server_3.sh; bash large_scale_test_server_3.sh #on the 3th server

bash large_scale_train_server_4.sh; bash large_scale_test_server_4.sh #on the 4th server

# For results: 
# 1. download src/output/run folder (for tensorboard viewing)
# 2. download src/output/result folder (info saved from test stage)
