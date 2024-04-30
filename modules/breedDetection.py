import torch
from pathlib import Path
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

class PlotImage:
    def __init__(self):
        pass

    def pred_image_class(model, filename, classifier):
        class_names = {
            'dog': ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'tan_coonhound', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa', 'coated_retriever', 'coated_retriever', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'],
            'cat': ['Abyssinian', 'Bengal', 'Exotic Shorthair', 'Persian', 'Ragdoll', 'Siamese', 'Siberian', 'Sphynx']
        }

        image = torchvision.io.read_image(str(Path(f'static/images/{filename}'))).type(torch.float32) / 255.
        image_transform = transforms.Compose([
            transforms.Resize(size=(244,244))
        ])
        image_transformed = image_transform(image)

        model.cpu()
        model.eval()
        with torch.inference_mode():
            image_pred = model(image_transformed.unsqueeze(0))
            image_pred_prob = torch.softmax(image_pred,dim=1)
            threshold = image_pred_prob.topk(1,dim=1).values.item()
            image_pred_label = torch.argmax(image_pred_prob,dim=1).cpu()
            
            if threshold >= 0.6:
                return f'{class_names[classifier][image_pred_label]}', f'{threshold*100:.2f}%'
            return f"could be a {class_names[classifier][image_pred_label]}",f"{threshold*100:.2f}%?"

class BreedModel:
    def __init__(self):
        pass
    
    def get_model(classifier):
        return torch.jit.load(Path(f'models/vgg_{classifier}.pth'))
    
    def classify_image(classifier, filename):
        model = BreedModel.get_model(classifier)
        return PlotImage.pred_image_class(model=model,filename=filename,classifier=classifier)
