from clip_cpp import Clip

## you can either pass repo_id or .gguf file
## you can type `clip-cpp-models` in your terminal to see what models are available for download
## in case you pass repo_id and it has more than .bin file
## it's recommended to specify which file to download with `model_file`
repo_id = 'mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K'
model_file = 'CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf'

model = Clip(
    model_path_or_repo_id=repo_id,
    model_file=model_file,
    verbosity=2
)

text_2encode = 'cat on a Turtle'

tokens = model.tokenize(text_2encode)
text_embed = model.encode_text(tokens)

## load and extract embeddings of an image from the disk
image_2encode = '/path/to/cat.jpg'
image_embed = model.load_preprocess_encode_image(image_2encode)

## calculate the similarity between the image and the text
score = model.calculate_similarity(text_embed, image_embed)

# Alternatively, you can just do:
# score = model.compare_text_and_image(text, image_path)

print(f"Similarity score: {score}")
