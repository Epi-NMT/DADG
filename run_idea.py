import torch
import Model_
import utils_
import numpy as np
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import time
from data_loaders_ import PACS_Loader, VLCS_Loader, OH_Loader
import argparse

pacs_datasets = ['P', 'A', 'C', 'S']
vlcs_datasets = ['VOL', 'LABELME', 'CALTECH', 'SUN']
office_home_datasets = ['art', 'clipart', 'product', 'real_world']
availabel_datasets = pacs_datasets + vlcs_datasets + office_home_datasets
dataset = ['PACS', 'VLCS', 'Office_Home']
model = ['resnet18', 'alexnet']

class idea_train():
    def __init__(self, flags):
        print('flags: ', flags, '\n')
        self.configure = flags
        self.device = torch.device('cuda:{}'.format(flags.gpu) if torch.cuda.is_available() else 'cpu')

        self.loss_fn = utils_.loss_fn()

        self.netE, self.netC, self.netD = Model_.nets(flags.model_name, flags.num_classes)

        self.netE, self.netC, self.netD = self.netE.to(self.device), \
                                          self.netC.to(self.device), \
                                          self.netD.to(self.device)

        self.netE_dup, self.netC_dup = self.netE.to(self.device), self.netC.to(self.device)

        self.opti = utils_.sgd(flags.model_name, list(self.netE.parameters())
                              + list(self.netC.parameters()), flags.lr)

        self.domain_opti = utils_.domain_sgd(flags.model_name, list(self.netE.parameters())
                                            + list(self.netD.parameters()), flags.domain_lr)

        print('Using device:', self.device, ', Using model:', flags.model_name, '\n')


    def unseen_test(self, flags):

        if flags.dataset == 'PACS':
            test_loader = PACS_Loader(flags).test_loader()

        if flags.dataset == 'VLCS':
            test_loader = VLCS_Loader(flags).test_loader()

        if flags.dataset == 'Office_Home':
            test_loader = OH_Loader(flags).test_loader()

        pred_test = np.array([])
        label_test = np.array([])
        with torch.no_grad():

            for i, (data, label) in enumerate(test_loader):
    
                data, label = Variable(data).to(self.device), \
                              Variable(label).to(self.device)
    
                label_test = np.append(label_test, label.data.cpu().numpy())
    
                predict = self.netC(self.netE(data))
    
    
                pred_label = torch.max(predict, 1)[1].data.cpu().numpy()
                pred_test = np.append(pred_test, pred_label)
    
            accuracy_test = accuracy_score(label_test, pred_test)

        return accuracy_test

    def train(self, flags):

        self.best_accuracy = 0
        self.netE.train()
        self.netC.train()
        self.netD.train()
        self.domain_loss_idx = np.array([])
        self.acc_idx = np.array([])
        for ite in range(flags.iteration):

            if flags.dataset == 'PACS':
                loader1, loader2, meta_loader, rand_number = PACS_Loader(flags).train_loaders()

            if flags.dataset == 'VLCS':
                loader1, loader2, meta_loader, rand_number = VLCS_Loader(flags).train_loaders()

            if flags.dataset == 'Office_Home':
                loader1, loader2, meta_loader, rand_number = OH_Loader(flags).train_loaders()

                if ite > int(flags.iteration * .7):
                    self.opti = utils_.sgd(flags.model_name, list(self.netE.parameters())
                                          + list(self.netC.parameters()), flags.lr/10)

            for domain_tr_iter, ((x1, y1), (x2, y2)) in enumerate(zip(loader1, loader2)):
                data, inputs = Variable(x1).to(self.device), \
                               Variable(x2).to(self.device)

                predict1 = self.netD(self.netE(data), ite, flags.iteration)
                predict2 = self.netD(self.netE(inputs), ite, flags.iteration)
                domain_loss = 0.5 * (- torch.mean(torch.log(predict1) + torch.log(1. - predict2)))
                self.domain_opti.zero_grad()
                domain_loss.backward(retain_graph=True)
                self.domain_opti.step()

                data, label = Variable(x1).to(self.device), \
                              Variable(y1).to(self.device)

                inputs, target = Variable(x2).to(self.device), \
                                 Variable(y2).to(self.device)

                prediction1 = self.netC(self.netE(data))
                prediction2 = self.netC(self.netE(inputs))

                label_cat = torch.cat((label, target), 0)
                predict_cat = torch.cat((prediction1, prediction2), 0)

                train_loss = self.loss_fn(predict_cat, label_cat)

                self.opti.zero_grad()
                train_loss.backward(retain_graph=True)
                self.opti.step()
                break

            for meta_test_iter, (x, y) in enumerate(meta_loader):

                data, label = Variable(x1).to(self.device), \
                              Variable(y1).to(self.device)

                predict = self.netC(self.netE(data))

                test_loss = self.loss_fn(predict, label)

                total_loss = test_loss

                self.netE.load_state_dict(self.netE_dup.state_dict())
                self.netC.load_state_dict(self.netC_dup.state_dict())

                self.opti.zero_grad()
                total_loss.backward()
                self.opti.step()

                self.netE_dup.load_state_dict(self.netE.state_dict())
                self.netC_dup.load_state_dict(self.netC.state_dict())
                break
            
            if (ite + 1) % flags.step_size == 0:
                target_acc = self.unseen_test(flags)

                print('iteration:', ite + 1,
                      ' | meta train loss:', train_loss.data.cpu().numpy(),
                      ' | val loss:', test_loss.data.cpu().numpy(),
                      ' | domain loss:', domain_loss.data.cpu().numpy(),
                      ' | acc:', target_acc
                      )

                if target_acc > self.best_accuracy:
    
                    self.best_accuracy = target_acc
                    
                    torch.save(self.netE, './Best_model/extractor_{}_{}.pth'.format(flags.model_name, flags.target))
                    torch.save(self.netC, './Best_model/classifier_{}_{}.pth'.format(flags.model_name, flags.target))
            
                    print('-------------Target domain accuracy:', self.best_accuracy)

        return self.best_accuracy

def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--dataset", choices=dataset, default='PACS',
                                  help="PACS / VLCS / Office-Home")
    train_arg_parser.add_argument("--iteration", type=int, default=3000,
                                  help="train how many iterations")
    train_arg_parser.add_argument("--test_time", type=int, default=1,
                                  help="train how many times")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=32,
                                  help="batch size for each domain in meta-train, total 64")
    train_arg_parser.add_argument("--meta_test_batchsz", type=int, default=32,
                                  help="batch size for meta test, default is 32")
    train_arg_parser.add_argument("--test_batchsz", type=int, default=128,
                                  help="batch size for testing, default is 128")
    train_arg_parser.add_argument("--model_name", choices=model, default='alexnet',
                                  help="resnet18 / alexnet")
    train_arg_parser.add_argument("--num_classes", type=int, default=7,
                                  help="number of classes")
    train_arg_parser.add_argument("--step_size", type=int, default=50,
                                  help="print information per step size")
    train_arg_parser.add_argument("--source", choices=availabel_datasets, default=['P', 'A', 'S'],
                                  help="source domains", nargs='+')
    train_arg_parser.add_argument("--target", choices=availabel_datasets, default='C',
                                  help="target domains")
    train_arg_parser.add_argument("--lr", type=float, default=5e-4,
                                  help='learning rate of the model')
    train_arg_parser.add_argument("--domain_lr", type=float, default=1e-5,
                                  help='domain learning rate')
    return train_arg_parser.parse_args()

def main():
    flags = args()
    model_ = idea_train(flags=flags)
    acc = model_.train(flags=flags)
    return acc

if __name__ == "__main__":

    for i in range(args().test_time):

        print('\n--------------------running {} time--------------------'.format(i+1), '\n')

        print('target domain:', args().target)

        start_t = time.time()

        final_acc = main()

        end_t = time.time()

        total_time = int((end_t - start_t) / 60)

        print('training time:', total_time, 'mins')
        print('training around', total_time * (args().test_time - i - 1), 'mins left')



        

