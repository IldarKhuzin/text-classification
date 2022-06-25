import keras
import numpy as np
from transformers import  BertConfig, BertTokenizerFast
import pandas as pd
import re

# labels_dict = {'red':1, 'green': 2, 'blue': 3}

# скачиваем модели
colors_model3=keras.models.load_model('/home/ildar/PycharmProjects/text_class_500/colors_model3')

# скачиваем датасет для проверки

test_df = pd.read_csv('/home/ildar/docet/классификация/df_color_corrected_3/df_col_val_500_cor_3.csv')

# функция очистки текста

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def color_predict(phrase):
    '''
    функция принимает фразу, прогоняет её по трем моделям и выдает её цвет
    :param phrase: phrase
    :return: color
    '''
    model_name = 'bert-base-uncased'
    config = BertConfig.from_pretrained(model_name)
    max_length = 32
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    test_token= tokenizer(
        text=phrase,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)

    res = colors_model3.predict(
        x={'input_ids': test_token['input_ids'], 'attention_mask': test_token['attention_mask']}, batch_size=32)

    print(res)

    res = np.argmax(res) + 1

    #labels_dict = {'red':1, 'green': 2, 'blue': 3}

    if res == 1:
        return 'red'
    elif res == 2:
        return 'green'
    elif res == 3:
        return 'blue'
    else:
        return 'no color'


wrong = 0
right = 0
for i in range(len(test_df.text)):
    print(test_df.text[i])
    phrase = clean_text(test_df.text[i])
    color = color_predict(phrase)
    print(f'real color: {test_df.labels[i]} ')
    print(f'predicted color: {color}')
    if test_df.labels[i] == color:
        right +=1
    else:
        wrong +=1
    print(f'count: right = {right}, wrong = {wrong}, accuracy = {round((right * 100)/(i+1))}')
    print('')
