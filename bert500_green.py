
from transformers import BertConfig, BertTokenizerFast, TFAutoModel
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

# гиперпараметры модели
epochs = 2
batch_size = 8
validation_split = 0.15
dropout_size = 0.3
path_to_df = '/home/ildar/docet/классификация/df_colors_500_corrected/df_col_500_cor.csv'

# считываем БД, убираем лишний столбец, перемешиваем и кодируем в get_dummies
# labels_dict = {'red':1, 'green': 2, 'blue': 3}
df= pd.read_csv(path_to_df)
df = df.sample(frac=1).reset_index(drop=True)
df = pd.get_dummies(df, columns=["labels"])
df = df.drop(['Unnamed: 0', 'labels_red', 'labels_blue'], axis=1)

nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
lm = WordNetLemmatizer()

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

def lemma(text):
    word_list = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lm.lemmatize(w) for w in word_list])
    return lemmatized_output

df['text'] = df['text'].map(lambda x : clean_text(x))

for i in range(len(df['text'])):
    df['text'][i] = lemma(df['text'][i])

train_sentences = df["text"].fillna("CVxTz").values
list_classes = ['labels_green']
train_y = df[list_classes].values

# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 64 # max 512

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
#config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)

input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
x = bert.bert(inputs)

print(x)

x2 =Dense(512, activation='relu')(x[1])
x3 = Dropout(dropout_size)(x2)
y =Dense(1, activation='sigmoid', name='outputs')(x3)

model = Model(inputs=inputs, outputs=y)

# Take a look at the model
print(model.summary())

optimizer = Adam(learning_rate=1e-5, decay=1e-6)
model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

# Tokenize the input
x = tokenizer(
    text=list(train_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length', # padding=True initial value,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

history = model.fit(
    x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    y={'outputs': train_y},
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=epochs)

model.save('colors_green500',save_format='tf')

