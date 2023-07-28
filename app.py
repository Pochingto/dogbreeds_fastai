import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('model2.pkl')

def get_labels(learn):
    labels = learn.dls.vocab
    labels = [label.replace("_", " ") for label in labels]
    labels[4] = "Alaskan malamute/ Siberian Husky" # indifferentiable
    return labels

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    labels = get_labels(learn)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

if __name__ == "__main__":
    title = "Dog Breed Classifier"
    description = "A dog breed classifier trained with fastai. Created as a demo for Gradio and HuggingFace Spaces."
    article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
    examples = ['dog1.jpg']
    interpretation='default'
    enable_queue=True

    gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3), title=title,description=description,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()
