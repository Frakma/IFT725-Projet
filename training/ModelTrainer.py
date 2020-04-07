import torch
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from typing import Callable, Type
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader

#we followed the same steps used in tp3
class ModelTrainer(object):
    def __init__(self, model, data_train, data_test,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size=1,
                 validation=None,
                 use_cuda=False):        
        
        
               
        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.validation=validation

        if self.validation is not None:
            self.data_train , self.data_validation = train_test_split(data_train, test_size=0.2, shuffle=False)
        else:
            self.data_train = data_train

        self.model = model        
        self.data_test = data_test
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer_factory(self.model)

        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}

    def train(self, num_epochs):

        # Initialize metrics container
        self.metric_values['train_loss'] = []
        self.metric_values['train_acc'] = []
        self.metric_values['val_loss'] = []
        self.metric_values['val_acc'] = []

        self.train_loader = DataLoader(self.data_train, batch_size, shuffle=True)


        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []

                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data_train is a list of [inputs, labels]
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    train_accuracies.append(self.accuracy(outputs, labels))

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
            
            # evaluate the model on validation data after each epoch
            self.metric_values['train_loss'].append(np.mean(train_losses))
            self.metric_values['train_acc'].append(np.mean(train_accuracies))
            if self.validation is not None:
                self.evaluate_on_validation_set()

        print('Finished Training')

    def accuracy(self, outputs, labels):
        acc=[]
        for predicted, label in zip(outputs, labels):
            m = tf.keras.metrics.Accuracy() 
            _ = m.update_state(predicted, label) 
            acc.append(m.result().numpy() )
        
        return np.mean(acc)

    def evaluate_on_validation_set(self):
        self.model.eval()

        val_loader = DataLoader(self.data_validation, batch_size, shuffle=True)
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                validation_losses.append(loss.item())

                validation_accuracies.append(self.accuracy(outputs, labels))
                validation_loss += loss.item()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        self.model.train()



    def evaluate_on_test_set(self):
        """
        Evaluate the model on test set
        return:
            Test Accuracy 
        """
        test_loader = DataLoader(self.data_test, batch_size, shuffle=True)
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                accuracies += self.accuracy(test_outputs, test_labels)
        print("Accuracy sur l'ensemble de test: {:05.3f} %".format(100 * accuracies / len(test_loader)))

    def plot_metrics(self):
        """
        Function that plots train and validation losses and accuracies after training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(epochs, self.metric_values['train_loss'], '-o', label='Training loss')
        ax1.plot(epochs, self.metric_values['val_loss'], '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(epochs, self.metric_values['train_acc'], '-o', label='Training accuracy')
        ax2.plot(epochs, self.metric_values['val_acc'], '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        f.savefig('fig.png')
        plt.show()


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f