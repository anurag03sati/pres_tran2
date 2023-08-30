
import numpy as np
import dill as dpickle
import pandas as pd
from utils import *


data2 = pd.read_csv('updateddf.csv')
from sklearn.model_selection import train_test_split

train, test = train_test_split(data2, test_size=0.2)
train, valid = train_test_split(train, test_size=0.2)


def load_decoder_inputs(decoder_np_vecs='transcription_train_vecs.npy'):
  vectorized_title = np.load(decoder_np_vecs)
  decoder_input_data=vectorized_title[:,:-1]#decoder input we dont need last word since using teacher enforcing
  decoder_target_data=vectorized_title[:,1:]

  print(f'Shape of decoder input: {decoder_input_data.shape}')
  print(f'Shape of decoder target: {decoder_target_data.shape}')
  return decoder_input_data,decoder_target_data


def load_encoder_inputs(encoder_np_vecs='prescription_train_vecs.npy'):
  vectorized_body = np.load(encoder_np_vecs)
  encoder_input_data=vectorized_body
  doc_length = encoder_input_data.shape[1]

  print(f'Shape of encoder input: {encoder_input_data.shape}')
  return encoder_input_data,doc_length

# cars=["audi","bmw","honda"]
# file="example.pkl"
# fileobj=open(file,'wb')
# pickle.dump(cars,fileobj)



def load_text_processor(fname='prescription_pp.dpkl'):
  with open(fname,'rb') as f:
    pp=dpickle.load(f)
  num_tokens=max(pp.id2token.keys())+1
  print(f'Size of vocabulary for {fname} : {num_tokens:,}')
  return num_tokens,pp

encoder_input_data, doc_length = load_encoder_inputs('prescription_train_vecs.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs('transcription_train_vecs.npy')

num_encoder_tokens, prescription_pp=load_text_processor('prescription_pp.dpkl')
num_decoder_tokens, transcription_pp=load_text_processor('transcription_pp.dpkl')



from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers

latent_dim = 300
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')

x = Embedding(num_encoder_tokens, latent_dim, name='Body-Word-Embedding', mask_zero=False)(encoder_inputs)
x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

_, state_h = GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)

encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)

decoder_inputs = Input(shape=(None,),
                       name='Decoder-Input')  # for teacher forcing thus pass decoder input to decoder model

dec_emb = Embedding(num_decoder_tokens, latent_dim, name='Decoder-Word-Embedding', mask_zero=False)(decoder_inputs)
dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

decoder_gru = GRU(latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU')
decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')
decoder_outputs = decoder_dense(x)

seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

seq2seq_Model.load_weights('minor_proj.epoch67-val5.24216.hdf5')  # using the best model among all epochs


def extract_encoder_model(model):
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model


def extract_decoder_model(model):
    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]

    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input])

    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
    decoder_model = Model([decoder_inputs, gru_inference_state_input], [dense_out, gru_state_out])
    return decoder_model


class Seq2Seq_Inference(object):
    def __init__(self, encoder_preprocessor, decoder_preprocessor, seq2seq_model):
        self.pp_prescription = encoder_preprocessor
        self.pp_transcription = decoder_preprocessor
        self.seq2seq_model = seq2seq_Model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len_transcription = self.pp_transcription.padding_maxlen
        self.nn = None
        self.rec_df = None

    def generate_transcription(self, raw_input_text, max_len_transcription=None):
        if max_len_transcription is None:
            max_len_transcription = self.default_max_len_transcription

        raw_tokenized = self.pp_prescription.transform([raw_input_text])
        body_encoding = self.encoder_model.predict(raw_tokenized)
        original_body_encoding = body_encoding
        state_value = np.array(self.pp_transcription.token2id['_start_']).reshape(1, 1)

        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict([state_value, body_encoding])
            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            pred_word_str = self.pp_transcription.id2token[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len_transcription:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            body_encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_body_encoding, ' '.join(decoded_sentence)

    def print_example(self, i, prescription_text, transcription_text, threshold):
        if i:
            print("================================================")
            print(f"Prescription Text : {prescription_text}")

            if transcription_text:
                print(f"Transcription (original) Text : {transcription_text}")

            emb, gen_transcription = self.generate_transcription(prescription_text)

            print(f"Machine generated Transcription : {gen_transcription}")

            if self.nn:
                n, d = self.nn.get_nns_by_vector(emb.flatten(), n=4, include_distances=True)
                neighbours = n[1:]
                dist = d[1:]

                if min(dist) <= threshold:
                    cols = ['prescription', 'transcription']
                    dfcopy = self.rec_df.iloc[neighbours][cols].copy(deep=True)
                    dfcopy['dist'] = dist
                    similar_issues_df = dfcopy.query(f'dist<={threshold}')

    def demo_model_predictions(self, n, df, threshold=1):
        prescription_text_list = df['prescription'].tolist()
        transcription_text_list = df['transcription'].tolist()
        demo_list = np.random.randint(low=1, high=len(prescription_text_list), size=n)

        for i in demo_list:
            self.print_example(i, prescription_text=prescription_text_list[i],
                               transcription_text=transcription_text_list[i], threshold=threshold)


seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=prescription_pp, decoder_preprocessor=transcription_pp,
                                seq2seq_model=seq2seq_Model)
seq2seq_inf.demo_model_predictions(n=50, df=test)

