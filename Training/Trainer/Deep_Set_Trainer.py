import random
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ROOT import TFile
from .Architectures.DeepSets.DeepSet import Model
from .DataStructures.ROOT_Dataset import ROOT_Dataset, collate
from .Analyzer import plot_ROC
import time

class Set_Agent:
    """Driver Class for deep sets

    Attributes:
        options (dict): configuration for the nn

    Methods:
        get_data_batches: get batched training and testing data from the data loaders
        train_and_test: convienience function for preparing training and testing the model
        save_agent: logs all the details of the training and saves it for future reference
    """

    def __init__(self, options):
        """Sets up a new agent, or loads a saved agent if training is being resumed.

        Args:
            options (dict): configuration options
        Returns:
            None
        """
        def _load_data(data_filename):
            """Reads the input data and sets up training and test data loaders.

            Args:
                data_filename: ROOT ntuple containing the relevant data
            Returns:
                train_loader, test_loader
            """
            # load data files
            t0 = time.time()
            print("Loading data")
            data_file = TFile(data_filename)
            data_tree = getattr(data_file, self.options["tree_name"])
            n_events = data_tree.GetEntries()
            data_file.Close()  # we want each ROOT_Dataset to open its own file and extract its own tree

            # perform class balancing

            print("Balancing classes")
            event_indices = np.array(range(n_events))
            full_dataset = ROOT_Dataset(data_filename, event_indices, self.options, shuffle_indices=False)
            # import pdb; pdb.set_trace()
            truth_values = [truth[0].bool() for _, _, truth in full_dataset]
            class_0_indices = list(event_indices[[i.item() for i in truth_values]])
            class_1_indices = list(event_indices[[not i.item() for i in truth_values]])
            n_each_class = min(len(class_0_indices), len(class_1_indices))
            random.shuffle(class_0_indices)
            random.shuffle(class_1_indices)
            balanced_event_indices = class_0_indices[:n_each_class] + class_1_indices[:n_each_class]
            n_balanced_events = len(balanced_event_indices)
            del full_dataset
            # split test and train

            print("Splitting and processing test and train events")
            random.shuffle(balanced_event_indices)
            n_training_events = int(self.options["training_split"] * n_balanced_events)
            train_event_indices = balanced_event_indices[:n_training_events]
            test_event_indices = balanced_event_indices[n_training_events:]

            train_set = ROOT_Dataset(data_filename, train_event_indices, self.options)
            test_set = ROOT_Dataset(data_filename, test_event_indices, self.options)

            # prepare the data loaders

            print("Prepping data loaders")
            train_loader = DataLoader(
                train_set,
                batch_size=self.options["batch_size"],
                collate_fn=collate,
                shuffle=True,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=self.options["batch_size"],
                collate_fn=collate,
                shuffle=True,
                drop_last=True,
            )

            return train_loader, test_loader

        self.options = options
        self.model = Model(self.options).to(self.options["device"])
        self.train_loader, self.test_loader = _load_data(self.options["input_data"])
        self.history_logger = SummaryWriter()

        # load previous state if training is resuming
        self.resumed_epoch_n = 0
        if self.options["continue_training"] is True:
            saved_agent = torch.load(self.options["model_path"])
            self.model.load_state_dict(saved_agent['model_state_dict'])
            self.model.optimizer.load_state_dict(saved_agent['optimizer_state_dict'])
            self.resumed_epoch_n = saved_agent['epoch']
        print("Model parameters:\n{}".format(self.model.parameters))

    def train_and_test(self, do_print=True):
        """Trains and tests the model.

        Args:
            do_print (bool, True by default): Specifies whether to print validation characteristics on each step
        Returns:
            None
        """
        def _train(Print=True):
            """Trains the model and saves training history to history logger.

            Args:
                Print (bool, True by default): Specifies whether to print training characteristics on each step
            Returns:
                None
            """
            for epoch_n in range(self.resumed_epoch_n, self.options["n_epochs"] + self.resumed_epoch_n):
                train_loss, train_acc, _, train_truth = self.model.do_train(self.train_loader)
                test_loss, test_acc, _, test_truth = self.model.do_eval(self.test_loader)
                self.history_logger.add_scalar("Accuracy/Train Accuracy", train_acc, epoch_n)
                self.history_logger.add_scalar("Accuracy/Test Accuracy", test_acc, epoch_n)
                self.history_logger.add_scalar("Loss/Train Loss", train_loss, epoch_n)
                self.history_logger.add_scalar("Loss/Test Loss", test_loss, epoch_n)
                for name, param in self.model.named_parameters():
                    self.history_logger.add_histogram(name, param.clone().cpu().data.cpu().numpy(), epoch_n)

                if Print:
                    print(
                        "Epoch: %03d, Train Loss: %0.4f, Train Acc: %0.4f, "
                        "Test Loss: %0.4f, Test Acc: %0.4f"
                        % (epoch_n, train_loss, train_acc, test_loss, test_acc)
                    )
                if (epoch_n) % 10 == 0:
                    torch.save({
                        'epoch': epoch_n,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.model.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                    }, self.options["model_path"])

        def _test():
            """Evaluates the model on testing batches and saves ROC curve to history logger."""
            _, _, test_raw_results, test_truth = self.model.do_eval(self.test_loader)
            ROC_fig = plot_ROC.plot_ROOT_ROC(self.options, test_raw_results, test_truth)
            self.history_logger.add_figure("ROC", ROC_fig)

        _train(do_print)
        _test()

    def save_agent(self):
        """saves the model and closes the TensorBoard SummaryWriter."""
        if not pathlib.Path(self.options["output_folder"]).exists():
            pathlib.Path(self.options["output_folder"]).mkdir(parents=True)
        # self.history_logger.export_scalars_to_json(self.options["output_folder"] + "/all_scalars.json")
        self.history_logger.close()


def train(options):
    """Driver function to load data, train model and save results.

    Args:
        options (dict) : configuration for the nn
    Returns:
         None
    """
    def _set_features(options):
        """Modifies options dictionary with branch name info."""
        data_file = TFile(options["input_data"])
        data_tree = getattr(data_file, options["tree_name"])
        options["branches"] = [i.GetName() for i in data_tree.GetListOfBranches()]
        options["baseline_features"] = [i for i in options["branches"] if i.startswith("baseline_")]
        options["lep_features"] = [i for i in options["branches"] if i.startswith("lep_")]
        options["trk_features"] = [i for i in options["branches"] if i.startswith("trk_")]
        options["n_lep_features"] = len(options["lep_features"])
        options["n_trk_features"] = len(options["trk_features"])
        data_file.Close()
        return options

    options = _set_features(options)
    agent = Set_Agent(options)
    agent.train_and_test()
    agent.save_agent()