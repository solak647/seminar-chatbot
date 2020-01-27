from data import get_data, get_data_onelabel
import pandas as pd
from simpletransformers.classification import ClassificationModel

#labels = ['cEXT','cNEU','cAGR','cCON','cOPN']
labels = ['cOPN']

models = [
    ['distilbert', 'distilbert-base-uncased', True],
    #['distilbert', 'distilbert-base-uncased-distilled-squad', True],
    #['roberta', 'roberta-base', False],
    #['roberta', 'roberta-large', False],
    #['albert', 'albert-base-v2', False],
    #['albert', 'albert-large-v2', False],
    #['camembert', 'camembert-base', False],
    #['xlnet', 'xlnet-base-cased', False],
    #['bert', 'bert-base-uncased', True],
]

epochs = [8]
learning_rates = [1e-5]

for label in labels:
    print("Training for label", label)
    training_dataset, testing_dataset = get_data_onelabel(label)
    # Class count
    count_class_0, count_class_1 = training_dataset.labels.value_counts()

    # Divide by class
    df_class_0 = training_dataset[training_dataset['labels'] == 0]
    df_class_1 = training_dataset[training_dataset['labels'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)
    training_dataset = pd.concat([df_class_0_under, df_class_1], axis=0)
    for model_name in models:
        for epoch in epochs:
            for learning_rate in learning_rates:
                dir_name = 'outputs/'+str(label)+'_'+str(model_name[1])+'_'+str(epoch)+'_'+str(learning_rate)+'_len256_v1'
                args = {
                    'output_dir': dir_name,
                    'reprocess_input_data': True,
                    'evaluate_during_training': True,
                    'num_train_epochs': epoch,
                    'learning_rate': learning_rate,
                    'do_lower_case': model_name[2],
                    'use_multiprocessing': True,
                    'overwrite_output_dir': True,
                    'use_early_stopping': True,
                    'silent': True,
                    'max_seq_length': 256,
                }
                model = ClassificationModel(model_name[0], model_name[1], num_labels=2, use_cuda=True, args=args)
                model.train_model(training_dataset, show_running_loss=True, eval_df=testing_dataset)
                result, model_outputs, wrong_predictions = model.eval_model(testing_dataset)

