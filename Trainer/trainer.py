import numpy as np
from Architectures.RNN import RNN
from DataStructures.HistoryData import HistoryData, LOSS, ACC, TRAIN, VALIDATION, TEST, BATCH, EPOCH
from DataStructures.LeptonTrackDataset import Torchdata, collate
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pickle as pkl

###############
# Tensorboard #
###############

writer = SummaryWriter()

##################
# Train and test #
##################


class RNN_Trainer:

    def __init__(self, options, leptons_with_tracks):
        self.options = options
        self.n_events = len(leptons_with_tracks)
        self.n_training_events = int(
            self.options['training_split'] * self.n_events)
        self.leptons_with_tracks = leptons_with_tracks
        self.options['n_track_features'] = len(
            self.leptons_with_tracks[0][1][0])

        # training results e.g. history[CLASS_LOSS][TRAIN][EPOCH]
        self.history = HistoryData()
        self.test_truth = []
        self.test_raw_results = []

    def prepare(self):
        # split train and test
        np.random.shuffle(self.leptons_with_tracks)
        self.training_events = \
            self.leptons_with_tracks[:self.n_training_events]
        self.test_events = self.leptons_with_tracks[self.n_training_events:]

        # prepare the generators
        self.train_set = Torchdata(self.training_events)
        self.test_set = Torchdata(self.test_events)

        # set up RNN
        self.rnn = RNN(self.options)

    def make_batch(self):

        training_loader = DataLoader(
            self.train_set, batch_size=self.options['batch_size'],
            collate_fn=collate, shuffle=True, drop_last=True)

        testing_loader = DataLoader(
            self.test_set, batch_size=self.options['batch_size'],
            collate_fn=collate, shuffle=True, drop_last=True)

        return training_loader, testing_loader

    def train(self, Print=True):
        train_loss = 0
        train_acc = 0
        for batch_n in range(self.options['n_batches']):
            training_batch, testing_batch = self.make_batch()
            train_loss, train_acc, _, _ = self.rnn.do_train(training_batch)
            test_loss, test_acc, _, _ = self.rnn.do_eval(testing_batch)
            self.history[LOSS][TRAIN][BATCH].append(train_loss)
            self.history[ACC][TRAIN][BATCH].append(train_acc)
            self.history[LOSS][TEST][BATCH].append(test_loss)
            self.history[ACC][TEST][BATCH].append(test_acc)
            writer.add_scalar('Accuracy/Train Accuracy', train_acc, batch_n)
            writer.add_scalar('Accuracy/Test Accuracy', test_acc, batch_n)
            writer.add_scalar('Loss/Train Loss', train_loss, batch_n)
            writer.add_scalar('Loss/Test Loss', test_loss, batch_n)
            if Print:
                print("Batch: %d, Train Loss: %0.4f, Train Acc: %0.4f, "
                      "Test Loss: %0.4f, Test Acc: %0.4f" % (
                          batch_n, train_loss, train_acc, test_loss, test_acc))
        return train_loss

    def test(self):
        # test_batch = []
        self.test_set.file.reshuffle()
        testing_loader = DataLoader(
            self.test_set, batch_size=self.options['batch_size'],
            collate_fn=collate, shuffle=True)
        _, _, self.test_raw_results, self.test_truth = self.rnn.do_eval(
            testing_loader)

    def train_and_test(self, do_print=True):
        '''Module to rum the execute the network'''
        self.prepare()
        loss = self.train(do_print)
        self.test()
        return loss

#################
# Main function #
#################


if __name__ == "__main__":

    # set options
    from Options.default_options import options

    # load data
    data_file = "../Data/lepton_track_data.pkl"
    leptons_with_tracks = pkl.load(open(data_file, 'rb'))
    options['lepton_size'] = len(leptons_with_tracks['lepton_labels'])
    options['track_size'] = len(leptons_with_tracks['track_labels'])

    # perform training
    lwt = list(
        zip(leptons_with_tracks['normed_leptons'],
            leptons_with_tracks['normed_tracks']))

    good_leptons = [i[leptons_with_tracks['lepton_labels'].index(
        'ptcone20')] > 0 for i in leptons_with_tracks['unnormed_leptons']]
    lwt = np.array(lwt)[good_leptons]
    RNN_trainer = RNN_Trainer(options, lwt)
    RNN_trainer.train_and_test()

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()