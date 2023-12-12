import json
import os
import subprocess

from collections import OrderedDict

# loads(your_json_string, object_pairs_hook=OrderedDict)

model_lists=['swin_tiny_patch4_window7_224','swin_small_patch4_window7_224','swin_base_patch4_window7_224',
'swinv2_tiny_patch4_window8_256','swinv2_small_patch4_window8_256','swinv2_base_patch4_window8_256','videoSwin.1x3x128x64x64']


if __name__ == '__main__':
    print('hello world')
    with open('v2_hk_swint_fp32_batch_1.json', 'r+') as fcc_file:  #v2_hk_swint_fp32_batch_1.json
        fcc_data = json.load(fcc_file, object_pairs_hook=OrderedDict)
        # print(fcc_data)
        print(json.dumps(fcc_data, indent=4))
        fcc_data['caseOwner'] = 'ning.zhang'
        model_type = 'int8'
        for model in model_lists:
            model_path = '/mercury/share/DLI/models/customer/hk/'+model+'.onnx'

            for batch in [1,2,8]:
                print(batch)
                if model_type == 'fp32':
                    fcc_data['quantize'] = "no"
                elif  model_type == 'fp16':
                    fcc_data['quantize'] = 'downcast'
                elif model_type == 'int8':
                    fcc_data['quantize'] = 'quantize'
                else:
                    print('model_type err!')
                    exit (-1)

                fcc_data['accuracy'] = model_type
                fcc_data['batch'] = batch
                fcc_data['maxBatch'] = 32
                fcc_data['modelDir'] = model_path
                name_new = model.replace('_', '-', 10)
                model_name=name_new + '-asyncMhaOpt'
                fcc_data['modelName'] = model_name # it must equal with the name in json

                case_dir = 'v2_hk_' + model_name+ '_' + model_type + '_batch_'+str(batch)
                case_json = case_dir + '.json'
                if not os.path.exists(case_dir):
                    os.mkdir(case_dir)

                jsonFile = open(case_dir+'/'+case_json, "w")
                jsonFile.write(json.dumps(fcc_data, indent=4))
                shellcmd = "cd ${cur_dir} && ../run_case.sh c++/hk/"+case_dir+'/'+case_json + '\n'
                with open('test.txt', mode = 'a') as f:
                    f.write(shellcmd)
                # print(f'shell cmd: {shellcmd}')







