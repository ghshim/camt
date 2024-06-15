'''This code is borrowed from https://github.com/facebookresearch/InferSent/'''
import os
import pickle
import argparse
import numpy as np

from sentence_transformers import SentenceTransformer

def encode(sentences, print_on=False):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    if print_on:
        for sentence, embedding in zip(sentences, embeddings):
            print("Sentence:", sentence)
            print("Embedding:", embedding)
            print("-------------------------")
    return embeddings

def main(args):
    filename = args.filename
    data = os.path.join(args.data_dir, filename)

    with open(data, 'rb') as input_file:
        data_list = pickle.load(input_file)
     
    sentences = [motion['label'] for data in data_list for motion in data['motions']]
    embeddings = encode(sentences)
   
    embedding_iter = iter(embeddings)
    sentence_iter = iter(sentences)
    for data in data_list:
        for motion in data['motions']:
            sentence = next(sentence_iter)
            assert motion['label'] == sentence
            embedding = next(embedding_iter)
            motion['sentence_embedding'] = embedding
        # print(data)
    
    output_path = os.path.join(args.data_dir, "prox_sentence_encoded.pkl")
    with open(output_path, 'wb') as output:
        pickle.dump(data_list, output)

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='extract-features',
        description='Extract features from pretrained Sentence Transformers')
    
    parser.add_argument('--data_dir', type=str, default='/home/gahyeon/Desktop/data/camt', help='the directory of prox dataset')
    parser.add_argument('--filename', type=str, default='prox.pkl', help='the directory of prox dataset')
    args = parser.parse_args()

    main(args)