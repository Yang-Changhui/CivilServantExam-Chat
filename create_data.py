import json

input_file = 'exam_instructions.jsonl'
output_file = 'exam_instructions_inernlm.jsonl'

system_value = '欢迎来到公务员答题大模型！我是你的专属答题助手，你可以向我提出问题并获得详细的答案。只要是关于公务员考试的问题，我都可以帮助你。'

output_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        conversation=[]
        data = json.loads(line.strip())
        data.pop('subject')
        data_keys=data.keys()
        input_value=''
        if 'textbox_answer' in data_keys and 'textbox_answer_analysis' in data_keys:
            output_value =f"答案为:{data['textbox_answer']}." +f"  答案分析:{data['textbox_answer_analysis']}"
            data.pop('textbox_answer')
            data.pop('textbox_answer_analysis')
        elif 'textbox_answer' in data_keys:
            output_value = f"答案为:{data['textbox_answer']}."
            data.pop('textbox_answer')
        

        for key in data_keys:
            if len(input_value)==0:
                key1=key.split('_')
                input_value = f"{key1[-1]}:{data[key]}."
            else:
                key1=key.split('_')
                input_value+=f" {key1[-1]}:{data[key]}."
        conversation.append({'system': system_value, 'input': input_value, 'output': output_value})
        output_data.append({"conversation":conversation})

with open(output_file, 'w', encoding='utf-8') as f:
    for item in output_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')