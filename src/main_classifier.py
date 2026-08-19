from config import GPT_2_SMALL
from finetuning.classification.spam.evaluation import SpamClassifierEvaluator
from finetuning.classification.spam.model import SpamClassifier
from finetuning.classification.spam.training import SmsSpamTrainer


if __name__ == '__main__':
    model = SpamClassifier(GPT_2_SMALL)

    print(f'{model.name}, fine-tuned for SMS spam classification')
    print(f'Number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Model size: {model.model_size}')
    print()

    trainer = SmsSpamTrainer()
    trainer.prepare_data()

    if model.state_file.exists():
        print('Loading model state from file...')
        model.load()
        print()
    else:
        print('Training the model... (takes about 6 minutes)')
        trainer.train_model(model)
        model.save()
        print()

    evaluator = SpamClassifierEvaluator(model)
    accuracy = evaluator.model_accuracy(trainer.test_data)
    print(f'Model accuracy: {accuracy * 100:.2f}%')
    print()

    message_1 = 'You are a winner!!! You have been specially selected to receive $1000 cash or a $2000 award!!!'
    is_spam = model.classify(message_1)
    print(message_1)
    print(f'Spam: {is_spam}')
    print()

    message_2 = "Hey! I just wanted to check if we're still on for dinner tonight? Let me know!"
    is_spam = model.classify(message_2)
    print(message_2)
    print(f'Spam: {is_spam}')
