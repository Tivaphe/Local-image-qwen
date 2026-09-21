"""Construit un faux YOLOv8-pose (mêmes entrées/sorties) encodant UNE pose connue.

Sert uniquement aux tests : il valide le chargement du modèle, l'inférence, le décodage
des sorties et le rendu du squelette, sans télécharger les 13 Mo du vrai yolov8n-pose.

Usage (facultatif, le fichier .onnx déjà construit est versionné) :
    pip install onnx && python tests/data/make_fake_onnx.py
"""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

SIZE, NK = 640, 17
CH = 4 + 1 + 3 * NK                      # 56

# valeurs encodées : une personne au centre, points articulaires connus
def build_values():
    v = np.zeros(CH, dtype="float32")
    v[0], v[1], v[2], v[3] = 320.0, 320.0, 128.0, 256.0   # cx, cy, w, h (repère 640x640 letterboxé)
    v[4] = 0.92                                            # confiance personnage
    kp = np.zeros((NK, 3), dtype="float32")
    for i in range(NK):
        kp[i] = (200.0 + 12 * i, 150.0 + 9 * i, 0.85)      # x, y, confiance du point
    v[5:] = kp.reshape(-1)
    return v

V = build_values()
w_conv = np.zeros((CH, 3, 1, 1), dtype="float32")          # poids nuls : sortie = biais
b_conv = V

nodes = [
    helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
    helper.make_node("Conv", ["pooled", "W", "B"], ["conv"], kernel_shape=[1, 1]),
    helper.make_node("Resize", ["conv", "roi", "scales", "sizes"], ["resized"], mode="nearest"),
    helper.make_node("Squeeze", ["resized"], ["output0"], axes=[2]),
]
graph = helper.make_graph(
    nodes, "fake-yolov8-pose",
    inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
    outputs=[helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, CH, 8400])],
    initializer=[
        numpy_helper.from_array(w_conv, "W"),
        numpy_helper.from_array(b_conv, "B"),
        numpy_helper.from_array(np.zeros((0,), dtype="float32"), "roi"),
        numpy_helper.from_array(np.zeros((0,), dtype="float32"), "scales"),
        numpy_helper.from_array(np.array([1, 1, 1, 8400], dtype="int64"), "sizes"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], producer_name="local-image-qwen-tests")
onnx.checker.check_model(model)
onnx.save(model, str(__import__("pathlib").Path(__file__).with_name("fake_yolov8n-pose.onnx")))
print("valeurs encodées : boîte (cx,cy,w,h) =", V[:4], "conf =", V[4], "| 1er point =", V[5:8])
