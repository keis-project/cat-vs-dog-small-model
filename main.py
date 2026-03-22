import os
import json
import numpy
from PIL import Image
import onnxruntime
import sys

def softmax(x):
    exp_x = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    return exp_x / numpy.sum(exp_x, axis=1, keepdims=True)

def read_image():
    make_dicts = {}
    path = "inputs/"
    filenames = os.listdir(path)

    if len(filenames) == 0:
        print("No images in inputs directory")
        sys.exit(1)


    for filename in filenames:
        make_dicts[filename] = {
            "answer": "",
            "cat_logit": 0.,
            "dog_logit": 0.
        }


    images = [numpy.array(Image.open(path+name).resize((32, 32)).convert('RGB'), dtype=numpy.float32) for name in filenames]


    images = numpy.stack(images, axis=0) / 255.0 * 2 - 1
    images = images.transpose((0, 3, 1, 2))


    return images.astype(numpy.float32), make_dicts

input_images, dicts = read_image()
session = onnxruntime.InferenceSession("model.onnx")

result = session.run(None, {"input": input_images})[0]
result = softmax(result)
for i, key in enumerate(dicts.keys()):
    r = "cat" if result[i][0] > result[i][1] else "dog"
    dicts[key]["answer"] = r
    dicts[key]["cat_logit"] = float(result[i][0])
    dicts[key]["dog_logit"] = float(result[i][1])

print(json.dumps(dicts, indent=2))
