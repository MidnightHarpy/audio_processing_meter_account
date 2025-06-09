from app.ai.models.entity_model import EntityModel

def main():
    model = EntityModel()
    texts, labels = model.preprocess_data('dataset.csv')
    model.train(texts, labels)
    model.save_model('model_directory')

if __name__ == "__main__":
    main()
