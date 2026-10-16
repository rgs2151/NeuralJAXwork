from tqdm import tqdm
import time

with tqdm(total=100, desc="Processing...", leave=True) as pbar:
    for i in range(100):
        # tqdm.write("To print something")
        time.sleep(0.1)
        pbar.set_description(f"Processing file {i}")
        pbar.update(1)


#    JAXsaw.Activation
#    JAXsaw.Layer
#    JAXsaw.Loss

#    JAXsaw.activations.Tanh
#    JAXsaw.layers.Dense
#    JAXsaw.losses.MSE
#    JAXsaw.losses.BinaryCrossEntropy