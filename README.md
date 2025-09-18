## Navoist with Knowledge Distillation

For whisper middle:
Normal model -> Encoder: 24 layers  -  Decoder: 24 layes - Parametrs: 763,857,920 (100%)
This model   -> Encoder: 24 layers  -  Decoder: 4  layes - Parametrs: 427,965,440 (56%)

The teory is that we main job is done by Encoder and even if we remove Decoder layers performance remains almost the same,
the diffrence will be in 1-2% (wer), as shown this article: https://arxiv.org/pdf/2311.00430 or in Whisper-X.

The model is >2x faster now! and if you apply faster-whisper transformation to a new model it can work real time)

"but the model is not ready now, as its wer is about 30-20%"

## How to use 

download the model.
'''bash
python3 load.py
'''

to run the server
'''bash
python3 server.py
'''

To test a model on audio of given path
you should edit faster.py there in thiis part:
'''bash
if __name__ == "__main__":
    t0 = time.time()
    path       = "./Usefull/rus1.wav" <- your audio file path
'''
