from finetuning.classification.spam.evaluator import SpamClassifierEvaluator
from finetuning.classification.spam.model import SpamClassifier
from finetuning.classification.spam.trainer import SmsSpamTrainer


if __name__ == '__main__':
    model = SpamClassifier()

    print('GPT-2 small (custom), fine-tuned for SMS spam classification')
    print(f'Number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Model size: {model.model_size}')

    trainer = SmsSpamTrainer()
    trainer.prepare_data()

    if model.state_file.exists():
        print('Loading model state from file...')
        model.load()
    else:
        print('Training the model... (takes about 6 minutes)')
        trainer.train_model(model)
        model.save()

    evaluator = SpamClassifierEvaluator(model)
    accuracy = evaluator.model_accuracy(trainer.test_data)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    message_1 = 'You are a winner!!! You have been specially selected to receive $1000 cash or a $2000 award!!!'
    is_spam = model.classify(message_1)
    print(message_1)
    print(f'Spam: {is_spam}')

    message_2 = "Hey! I just wanted to check if we're still on for dinner tonight? Let me know!"
    is_spam = model.classify(message_2)
    print(message_2)
    print(f'Spam: {is_spam}')
