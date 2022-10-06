import os

import elq.main_dense as main_dense
import argparse
from flask import Flask, request, jsonify, json



class Service:
    def __init__(self) -> None:
        """
        加载ELQ模型
        """
        models_path = "models/"

        ## config for elq
        config = {
            "interactive": False,
            "biencoder_model": models_path+"elq_wiki_large.bin",
            "biencoder_config": models_path+"elq_large_params.txt",
            "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "output_path": "logs/", # logging directory
            "faiss_index": "hnsw",
            "index_path": models_path+"faiss_hnsw_index.pkl",
            "num_cand_mentions": 10,
            "num_cand_entities": 10,
            "threshold_type": "joint",
            "threshold": -10,
        }

        self.args = argparse.Namespace(**config)

        print('Load models...')
        self.models = main_dense.load_models(self.args, logger=None)
        print('Models loaded')

        data_to_link = [{
                            "id": 0,
                            "text": "paris is capital of which country?".lower(),
                        },
                        {
                            "id": 1,
                            "text": "paris is great granddaughter of whom?".lower(),
                        },
                        {
                            "id": 2,
                            "text": "who discovered o in the periodic table?".lower(),
                        },]

        predictions = main_dense.run(self.args, None, *self.models, test_data=data_to_link)
        print('Initial predictions:',predictions)

    def predict(self,question):
        question_to_link = [{"id":0,"text":question}]
        res = main_dense.run(self.args,None,*self.models,test_data=question_to_link)
        #print(type(self.models))
        res[0]['dbpedia_ids'] = [[self.models[-1][res[0]['pred_triples'][i][0][j]] for j in range(len(res[0]['pred_triples'][i][0]))] for i in range(len(res[0]['pred_triples']))]
        return res
        
app = Flask(__name__)


@ app.route('/entity_linking', methods=['POST'])
def relation_detection_service():
    """
    params['question']: natural language question
    """
    if request.method == 'POST':
        decoded_data = request.data.decode('utf-8')
        params = json.loads(decoded_data)
        question = params['question']
        res = service.predict(question)
        print("res:"+str(res))
        return jsonify({'detection_res': res})


if __name__ == "__main__":

    # not using GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    service = Service()
    
    # initialize
    result = service.predict('which boxing stance is used by michael tyson?')
    print('result:' + str(result))

    #print(type(service.models))
    # service.close()

    # 0.0.0.0 makes it externally visible
    app.run(host="0.0.0.0", port=5688, debug=False)
