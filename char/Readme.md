Simple character-level model. Pretrained models are provided for:

* c64 (caffe64)
* intel (the intel instruction set reference)
* moby (*Moby Dick*, as found on Project Gutenberg)
* title (Arxiv paper titles, courtesy of Daniel Maturana)

Try:
    ./charmodel.py train txtfilename numTrainSamples modelPath
        writes a .bin file with the network, and .txt file with the
        dictionary
    ./charmodel.py sample modelPath numGenSamples temperature
        loads the model, prompts you for seeding text and starts. The softmax
        output is exponentiated according to the temperature parameter. High 
        values produce the most likely text, low values act more uniformly.
