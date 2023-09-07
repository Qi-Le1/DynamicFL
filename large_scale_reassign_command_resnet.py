def main():
    # run_file = open('./{}.txt'.format(f'large_scale_train'), 'w')
    # print(run_file)

    # run_file = open('./{}.txt'.format(f'large_scale_train'), 'r')
    # print(run_file)
    # print(run_file.read())
    # print('----')
    # for i in run_file:
    #     print(i)

    # import os
    # # print (os.path.exists('./{}.txt'.format(f'large_scale_train')))
    # # with open('./{}.txt'.format(f'large_scale_train'), 'r') as f:
    # #     ff = f.read()
    # #     print('ff', ff)
    # # return
    run = ['train', 'test']
    file = ['server_4', 'server_5']

    train_file_lists = [f'{run[0]}_{file[i]}' for i in range(len(file))]
    test_file_lists = [f'{run[1]}_{file[i]}' for i in range(len(file))]

    print(f'train_file_lists: {train_file_lists}')

    long_key_word = 'resnet'
    each_node_length = 12
    try:
        run_file = open('./{}.txt'.format(f'large_scale_train_server_commands_temp'), 'r')
        train_max_commands = run_file.readlines()

        # if len(command_group) == each_node_length:
        #     print('ceshiyixia', train_file_lists[index%(len(train_file_lists))])
        #     run_file = open('./{}.sh'.format(f'large_scale_{train_file_lists[index%(len(train_file_lists))]}'), 'a')
        #     run_file.write(''.join(command_group))
        #     run_file.close()
        #     # print(f'command_group:{command_group}')
        #     command_group = []
        #     index += 1

        index = 0
        command_group = []
        command_group_6_1 = []
        # for command in train_max_commands:
            # if 'wait' in command:
            #     print('yesss')
            #     continue
            # if long_key_word in command:
            #     continue
            # command_group.append(command)
        
        for command in train_max_commands:
            if 'wait' in command:
                continue
            if long_key_word not in command:
                continue            
            command_group.append(command)
        
        # print('command_group', command_group)
        start = 0
        end = min(len(command_group), each_node_length)
        while start < len(command_group):
            print('ceshiyixia', train_file_lists[index%(len(train_file_lists))])
            run_file = open('./{}.sh'.format(f'large_scale_{train_file_lists[index%(len(train_file_lists))]}'), 'a')
            if index < len(train_file_lists):
                run_file.write('#!/bin/bash\n')
            run_file.write(''.join(command_group[start:end]))
            run_file.write('wait\n')
            run_file.close()
            start = end
            end = min(len(command_group), end+each_node_length)
            index += 1

    except Exception as e:
        print(e)

    try:
        run_file = open('./{}.txt'.format(f'large_scale_test_server_commands_temp'), 'r')
        test_max_commands = run_file.readlines()

        index = 0
        command_group = []
        command_group_6_1 = []
        # for command in test_max_commands:
            # if 'wait' in command:
            #     continue
            # if long_key_word in command:
            #     continue
            # command_group.append(command)
        
        for command in test_max_commands:
            if 'wait' in command:
                continue
            if long_key_word not in command:
                continue            
            command_group.append(command)
        
        start = 0
        end = min(len(command_group), each_node_length)
        while start < len(command_group):
            print('ceshiyixia', test_file_lists[index%(len(test_file_lists))])
            run_file = open('./{}.sh'.format(f'large_scale_{test_file_lists[index%(len(test_file_lists))]}'), 'a')
            if index < len(test_file_lists):
                run_file.write('#!/bin/bash\n')
            run_file.write(''.join(command_group[start:end]))
            run_file.write('wait\n')
            run_file.close()
            start = end
            end = min(len(command_group), end+each_node_length)
            index += 1
    except Exception as e:
        print(e)


    # try:
    #     run_file = open('./{}.txt'.format(f'pre_run_large_scale_train'), 'r')
    #     train_max_commands = run_file.readlines()

    #     index = 0
    #     command_group = []
    #     for command in train_max_commands:
    #         command_group.append(command)
    #         if len(command_group) == 5:
    #             print('ceshiyixia', train_file_lists[index%(len(train_file_lists))])
    #             run_file = open('./{}.sh'.format(f'pre_run_large_scale_{train_file_lists[index%(len(train_file_lists))]}'), 'a')
    #             run_file.write(''.join(command_group))
    #             run_file.close()
    #             # print(f'command_group:{command_group}')
    #             command_group = []
    #             index += 1
    # except Exception as e:
    #     print(e)

    # try:
    #     run_file = open('./{}.txt'.format(f'pre_run_large_scale_test'), 'r')
    #     test_max_commands = run_file.readlines()

    #     index = 0
    #     command_group = []
    #     for command in test_max_commands:
    #         command_group.append(command)
    #         if len(command_group) == 5:
    #             print('ceshiyixia', test_file_lists[index%(len(test_file_lists))])
    #             run_file = open('./{}.sh'.format(f'pre_run_large_scale_{test_file_lists[index%(len(test_file_lists))]}'), 'a')
    #             run_file.write(''.join(command_group))
    #             run_file.close()
    #             # print(f'command_group:{command_group}')
    #             command_group = []
    #             index += 1
    # except Exception as e:
    #     print(e)
    
if __name__ == '__main__':
    main()
