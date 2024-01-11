import json
import matplotlib.pyplot as plt
import numpy as np

def extract_out_smooth_scales(json_data):
    out_smooth_scales_per_layer = {}

    for layer_id, values in json_data.items():
        for entry in values:
            if "out_smooth_scale" in entry:
                out_smooth_scales_per_layer[layer_id] = entry["out_smooth_scale"]

    return out_smooth_scales_per_layer

def main():
    with open('smooth_scales_and_shifts.json', 'r') as json_file:
      json_data = json.load(json_file)

    out_smooth_scales_per_layer = extract_out_smooth_scales(json_data)

    for layer_id, out_smooth_scale in out_smooth_scales_per_layer.items():
      x = x = np.arange(len(out_smooth_scale))
      plt.plot(x, out_smooth_scale, color="blue")
      
      plt.xlabel("Index")
      plt.ylabel("Value")
      plt.title("Plotting an array in Python")

      plt.show()

      breakpoint()
      print(f"Layer {layer_id}: out_smooth_scale = {out_smooth_scale}")

if __name__ == "__main__":
    main()
